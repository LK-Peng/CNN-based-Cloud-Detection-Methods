import torch
import os
import time
import json
import random
import numpy as np
from osgeo import gdal
from tqdm import tqdm
from torch.utils.data import DataLoader
from matplotlib.colors import LinearSegmentedColormap
from torchvision import transforms
from captum.attr import Saliency

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from dataloaders.dataset import RSERFSet
from model import get_network
from interpretation.calculate_erf import calculate_erf
from utils.img_saver import save_img
from config import get_config_erf


class ERF(object):
    def __init__(self, args):
        self.args = args

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        erf_set = RSERFSet(args)
        self.erf_loader = DataLoader(erf_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        # Define network
        model = get_network(args)

        # count parameters
        param_count = 0
        for param in model.parameters():
            param_count += param.view(-1).size()[0]
        print('Total parameters: {}M ({})'.format(param_count / 1e6, param_count))
        self.model = model

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            self.model = self.model.cuda()

    def cal_erf(self, load_path):
        # load
        checkpoint = torch.load(load_path)
        if self.args.cuda:
            self.model.module.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        sl = Saliency(self.model)

        tbar = tqdm(self.erf_loader, desc='\r')
        num_pixel = len(self.erf_loader.dataset)  # pixel num
        erf_img = dict()
        out_path = os.path.join(self.args.out_path, os.path.split(load_path)[-1])
        if not os.path.exists(out_path):
            os.mkdir(out_path)

        for i, sample in enumerate(tbar):
            image, targets = sample['image'], sample['label'].numpy()
            if self.args.cuda:
                image = image.cuda()
            if self.args.max_pro:
                with torch.no_grad():
                    output = self.model(image)
                pred = output.data.cpu().numpy()
                pred = np.argmax(pred, axis=1).astype(np.uint8)
                targets_tup = [(pred[i, targets[i][0], targets[i][1]], targets[i][0], targets[i][1]) for i in
                               range(len(targets))]
            else:
                targets_tup = [(1, hw[0], hw[1]) for hw in targets]
            image.requires_grad = True
            sl_attr = sl.attribute(image, target=targets_tup, abs=False)
            sl_img = sl_attr.data.cpu().numpy()

            b, _, _, _ = sl_img.shape
            for ib in range(b):
                filename = os.path.split(self.erf_loader.dataset.img_files[i * self.args.batch_size + ib])[-1]
                filename = filename.split(sep='.')[0] + '_h%.3d' % targets[ib][0] + '_w%.3d' % targets[ib][1] + '.tif'
                erf_img[filename] = calculate_erf(sl_img[ib, :], targets[ib, 0], targets[ib, 1])
                if self.args.save_img:  # and i % 100 == 0
                    save_img(sl_img[ib, :], os.path.join(out_path, filename))

        with open(os.path.join(self.args.out_path, os.path.split(load_path)[-1] + '.json'), 'w') as f:
            json.dump(erf_img, f, indent=4)

        print('ERF:')
        print('[numPixels: %5d]' % num_pixel)
        print("mean ERF:{}, std ERF:{}".format(np.mean(list(erf_img.values())), np.std(list(erf_img.values()))))


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
            args = get_config_erf('UNet-dilation')
            args.dilation = int(net_name.split(sep='-')[0][-1])
        else:
            args = get_config_erf('-'.join(net_name.split(sep='-')[0:-1]))
        args.seed = int(net_name.split(sep='-')[-1][4:])
        if 'MSCFF' in args.net or 'DeeplabV3Plus' in args.net:
            args.batch_size = 6
        
        # define parameters files
        args.load_paths = [load_roots[net_name]]

        # define output path
        args.out_path = os.path.join('./erf', net_name)
        if not os.path.exists(args.out_path):
            os.mkdir(args.out_path)

        print(args)

        torch.manual_seed(args.seed)  # set seed for the CPU
        np.random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # erf = ERF(args)
        for load_path in args.load_paths:
            erf = ERF(args)
            erf.cal_erf(load_path)
            del erf

        print('Using {}s!'.format(time.time() - start1))


if __name__ == '__main__':
    start = time.time()
    main()
    print('Using {}s!'.format(time.time() - start))