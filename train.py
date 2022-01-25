import os
import time
import numpy as np
import torch
from tqdm import tqdm

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from dataloaders import make_data_loader
from model import get_network
from model.deeplab.sync_batchnorm.replicate import patch_replication_callback
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.tracker import Tracker
from utils.metrics import Evaluator
from config import get_config_tr


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        # Define Tracker
        self.tracker = Tracker(run_directory=self.saver.experiment_dir)
        
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader = make_data_loader(args, **kwargs)
        self.nclass = args.num_classes

        # Define network
        model = get_network(args)

        # count parameters
        param_count = 0
        for param in model.parameters():
            param_count += param.view(-1).size()[0]
        print('Total parameters: {}M ({})'.format(param_count / 1e6, param_count))

        # the version of torch on GPU (windows) doesn't support the operation
        self.writer.add_graph(model, input_to_model=torch.zeros((args.batch_size, args.in_channels, 256, 256)))

        if args.net == 'DeeplabV3Plus':
            train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                            {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]
        else:
            train_params = [{'params': model.parameters(), 'lr': args.lr}]

        # Define Optimizer
        optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)

        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(os.path.split(args.train_list)[0], args.dataset+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(os.path.split(args.train_list)[0],
                                                  args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer
        
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        if args.net == 'DeeplabV3Plus':
            # "LR_Scheduler (step)" is the same as "torch.optim.lr_scheduler.StepLR"
            self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(self.train_loader))
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            if args.net == 'DeeplabV3Plus':
                patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

        # save untrained model
        is_best = False
        self.saver.save_checkpoint({
            'epoch': -1,
            'state_dict': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_pred': 0.0,
        }, is_best, filename='model_untrained.pth.tar')

    def training(self, epoch):

        # reset tracker
        self.tracker.begin_epoch()

        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_batch_tr = len(self.train_loader)  # iter/batch num
        num_img_tr = len(self.train_loader.dataset)  # image num
        for i, sample in enumerate(tbar):
            if self.args.net != 'RSNet':
                image, target = sample['image'], sample['label']
            else:
                image, target = sample['image'], \
                                sample['label'][:, self.args.top_crop: self.args.h - self.args.bottom_crop,
                                                self.args.left_crop: self.args.w - self.args.right_crop]
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            if self.args.net == 'DeeplabV3Plus':
                self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, target)
            loss.backward()
            if self.args.net != 'DeeplabV3Plus':
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
            train_loss += loss.item()
            # if (i + 1) % self.args.loss_interval == 0:
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_batch_tr * epoch)
            self.writer.add_scalar('train/lr_iter', self.optimizer.param_groups[0]['lr'], i + num_batch_tr * epoch)

        if self.args.net != 'DeeplabV3Plus':
            self.scheduler.step()

        train_loss = train_loss / num_img_tr
        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)  # train_loss / (i + 1)
        self.writer.add_scalar('train/lr_epoch', self.optimizer.param_groups[0]['lr'], epoch)
        self.tracker.train_epoch(epoch, train_loss, self.optimizer.param_groups[0]['lr'])  # track performance
        if self.args.net == 'DeeplabV3Plus':
            for name, weight in self.model.module.aspp.named_parameters():
                self.writer.add_histogram(f'train/{name}', weight, epoch)
                self.writer.add_histogram(f'train/{name}.grad', weight.grad, epoch)
            for name, weight in self.model.module.decoder.named_parameters():
                self.writer.add_histogram(f'train/{name}', weight, epoch)
                self.writer.add_histogram(f'train/{name}.grad', weight.grad, epoch)
        else:
            for name, weight in self.model.module.outc.named_parameters():
                self.writer.add_histogram(f'train/{name}', weight, epoch)
                self.writer.add_histogram(f'train/{name}.grad', weight.grad, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, num_img_tr))
        print('Loss: %.3f' % train_loss)

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        num_img_val = len(self.val_loader.dataset)  # image num
        val_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            val_loss += loss.item()
            # if (i + 1) % self.args.loss_interval == 0:
            tbar.set_description('Validation loss: %.3f' % (val_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1).astype(np.uint8)

            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        val_loss = val_loss / num_img_val
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        precision = self.evaluator.Precision()
        recall = self.evaluator.Recall()
        f_score = self.evaluator.F_score()
        self.writer.add_scalar('val/total_loss_epoch', val_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        self.writer.add_scalar('val/precision', precision, epoch)
        self.writer.add_scalar('val/recall', recall, epoch)
        self.writer.add_scalar('val/f_score', f_score, epoch)
        self.tracker.val_epoch(epoch, val_loss, Acc, Acc_class, mIoU, FWIoU)  # track performance
        self.tracker.end_epoch()
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, num_img_val))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print("precision:{}, recall:{}, f_score:{}".format(precision, recall, f_score))
        print('Loss: %.3f' % val_loss)

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best, filename='checkpoint-e' + str(epoch) + '-MIoU{:.3f}'.format(new_pred) + '.pth.tar')
        if self.args.save_epoch:
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best, filename='checkpoint-e' + str(epoch) + '-MIoU{:.3f}'.format(new_pred) + '.pth.tar')


def main():
    # choices=['DeeplabV3Plus', 'MFCNN', 'MSCFF', 'MUNet', 'TLNet', 'UNet', 'UNet-3', 'UNet-2', 'UNet-1', 'UNet-dilation', 'UNetS3', 'UNetS2', 'UNetS1']
    args = get_config_tr('TLNet')
    print(args)
    torch.manual_seed(args.seed)  # set seed for the CPU
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        start = time.time()
        trainer.training(epoch)  # train
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)  # validation
        print('Epoch: {}, Time: {}s!'.format(epoch, time.time() - start))

    trainer.writer.close()


if __name__ == "__main__":
    time1 = time.time()
    main()
    print('Time: {}s!'.format(time.time() - time1))
    print(time.strftime("%Y-%m-%d %H:%M:%S"))
