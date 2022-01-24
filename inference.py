import os
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from multiprocessing import Pool

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from dataloaders.dataset import RSSet
from model import get_network
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.metrics import Evaluator, BoundaryEvaluator
from utils.img_saver import save_img
from config import get_config_test


class Inference(object):
    def __init__(self, args):
        self.args = args

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        if args.no_gt:
            test_set = RSSet(args, split='test')
            self.test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        else:
            args.val_list, args.val_root = args.test_list, args.test_root
            test_set = RSSet(args, split='val')
            self.test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

            # Define Criterion
            # whether to use class balanced weights
            if args.use_balanced_weights:
                classes_weights_path = os.path.join(os.path.split(args.train_list)[0],
                                                    args.dataset + '_classes_weights.npy')
                if os.path.isfile(classes_weights_path):
                    weight = np.load(classes_weights_path)
                else:
                    weight = calculate_weigths_labels(os.path.split(args.train_list)[0],
                                                      args.dataset, self.train_loader, self.nclass)
                weight = torch.from_numpy(weight.astype(np.float32))
            else:
                weight = None
            self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)

        self.nclass = args.num_classes

        # Define network
        model = get_network(args)

        # count parameters
        param_count = 0
        for param in model.parameters():
            param_count += param.view(-1).size()[0]
        print('Total parameters: {}M ({})'.format(param_count / 1e6, param_count))
        self.model = model

        # define multiprocess
        if args.num_proc:
            self.p = Pool(processes=args.num_proc)
        else:
            self.p = None

        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        self.boundaryevaluator_3 = BoundaryEvaluator(self.nclass, self.p, self.args.num_proc, bound_th=3)
        self.boundaryevaluator_5 = BoundaryEvaluator(self.nclass, self.p, self.args.num_proc, bound_th=5)

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            self.model = self.model.cuda()

        # define dict to save metric
        self.metric_dct = dict()

    def test(self, load_path):
        # load
        checkpoint = torch.load(load_path)
        if self.args.cuda:
            self.model.module.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()

        tbar = tqdm(self.test_loader, desc='\r')
        num_img_val = len(self.test_loader.dataset)  # image num
        out_path = os.path.join(self.args.out_path, os.path.split(load_path)[-1])
        if not os.path.exists(out_path):
            os.mkdir(out_path)

        val_loss = 0.0
        self.evaluator.reset()
        self.boundaryevaluator_3.reset()
        self.boundaryevaluator_5.reset()
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            if not self.args.no_gt:  # ground truth is available
                loss = self.criterion(output, target)
                val_loss += loss.item()
                tbar.set_description('Validation loss: %.3f' % (val_loss / (i + 1)))
            pred, target = output.data.cpu().numpy(), target.cpu().numpy()
            pred = np.argmax(pred, axis=1).astype(np.uint8)

            if not self.args.no_gt:  # ground truth is available
                time.sleep(0.1)
                self.evaluator.add_batch(target, pred)
                self.boundaryevaluator_3.add_batch(target, pred)
                self.boundaryevaluator_5.add_batch(target, pred)

            if self.args.save_img:
                c, _, _ = pred.shape
                for ic in range(c):
                    filename = os.path.split(self.test_loader.dataset.img_files[i * self.args.batch_size + ic])[-1]
                    save_img(pred[ic, :, :], os.path.join(out_path, filename))

        self.metric_dct = {
            'PA': self.evaluator.Pixel_Accuracy(),
            'MPA': self.evaluator.Pixel_Accuracy_Class(),
            'MIoU': self.evaluator.Mean_Intersection_over_Union(),
            'FWIoU': self.evaluator.Frequency_Weighted_Intersection_over_Union(),
            'Precision': self.evaluator.Precision(),
            'Recall': self.evaluator.Recall(),
            'F1': self.evaluator.F_score(),
            'F_boundary_3': self.boundaryevaluator_3.F_score_boundary().tolist(),
            'Pr_boundary_3': self.boundaryevaluator_3.Precision_boundary().tolist(),
            'Re_boundary_3': self.boundaryevaluator_3.Recall_boundary().tolist(),
            'F_boundary_5': self.boundaryevaluator_5.F_score_boundary().tolist(),
            'Pr_boundary_5': self.boundaryevaluator_5.Precision_boundary().tolist(),
            'Re_boundary_5': self.boundaryevaluator_5.Recall_boundary().tolist(),
            'loss': val_loss / num_img_val,
        }

        print('Validation:')
        print('[numImages: %5d]' % num_img_val)
        print('Loss: %.3f' % val_loss)
        print(self.metric_dct)


def main():

    load_roots = {
        'DeeplabV3Plus-seed1': './ckp/DeeplabV3Plus-seed1.pth.tar',
        'DeeplabV3Plus-seed2': './ckp/DeeplabV3Plus-seed2.pth.tar',
        'DeeplabV3Plus-seed3': './ckp/DeeplabV3Plus-seed3.pth.tar',
        'DeeplabV3Plus-seed4': './ckp/DeeplabV3Plus-seed4.pth.tar',
    }

    net_names = [
        'DeeplabV3Plus-seed1', 'DeeplabV3Plus-seed2', 'DeeplabV3Plus-seed3', 'DeeplabV3Plus-seed4',
    ]

    for net_name in net_names:
        print('Using model {}'.format(net_name))
        start1 = time.time()

        if 'dilation' in net_name:
            args = get_config_test('UNet-dilation')
            args.dilation = int(net_name.split(sep='-')[0][-1])
        else:
            args = get_config_test('-'.join(net_name.split(sep='-')[0:-1]))
        args.seed = int(net_name.split(sep='-')[-1][4:])
        if 'MSCFF' in args.net or 'Deeplab' in args.net:
            args.batch_size = 256

        # define parameters files
        args.load_paths = [load_roots[net_name]]

        # define output path
        args.out_path = os.path.join('./inference', net_name)
        if not os.path.exists(args.out_path):
            os.mkdir(args.out_path)

        print(args)

        torch.manual_seed(args.seed)  # set seed for the CPU
        np.random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        for load_path in args.load_paths:
            print(load_path)
            start2 = time.time()
            inference = Inference(args)
            inference.test(load_path)
            with open(os.path.join(args.out_path, '{}-pixel.json'.format(os.path.split(load_path)[-1])), 'w') as f:
                json.dump(inference.metric_dct, f, indent=4)
            inference.p.close()
            del inference
            print('One parameter file using {}s!'.format(time.time() - start2))
        
        print('All parameter file using {}s!'.format(time.time() - start1))


if __name__ == '__main__':
    start = time.time()
    main()
    print('Using {}s!'.format(time.time() - start))

