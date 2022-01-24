import os
import json
import time
import argparse
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from torch.utils.data import DataLoader

from dataloaders.dataset import MaskSet
from utils.metrics import Evaluator, BoundaryEvaluator


class Comparator(object):
    def __init__(self, args):
        self.args = args

        # define dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        dataset = MaskSet(args)
        self.mask_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

        # define multiprocess
        if args.num_proc:
            self.p = Pool(processes=args.num_proc)
        else:
            self.p = None

        # Define Evaluator
        self.evaluator = Evaluator(self.args.num_classes)
        self.boundaryevaluator_3 = BoundaryEvaluator(self.args.num_classes, self.p, self.args.num_proc, bound_th=3)
        self.boundaryevaluator_5 = BoundaryEvaluator(self.args.num_classes, self.p, self.args.num_proc, bound_th=5)

    def cal_metric(self):
        tbar = tqdm(self.mask_loader, desc='\r')
        num_mask = len(self.mask_loader.dataset)
        print('numImages: {}'.format(num_mask))
        # metric_img = dict()
        for i, sample in enumerate(tbar):
            gt_mask, pre_mask = sample['gt'].numpy(), sample['pre'].numpy()

            self.evaluator.add_batch(gt_mask, pre_mask)
            self.boundaryevaluator_3.add_batch(gt_mask, pre_mask)
            self.boundaryevaluator_5.add_batch(gt_mask, pre_mask)

        metric_dct = {
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
        }
        with open(self.args.out_file, 'w') as f:
            json.dump(metric_dct, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare two mask')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--batch-size', type=int, default=24,
                        metavar='N', help='input batch size for comparison (default: auto)')
    parser.add_argument('--pre-root', type=str,
                        default=None,
                        help='mask root of prediction')
    parser.add_argument('--gt-root', type=str,
                        default='./example/test/Masks',
                        help='mask root of ground truth')
    parser.add_argument('--merge-class', action='store_true', default=True,
                        help='if merge class in ground truth')
    parser.add_argument('--num-classes', type=int, default=2,
                        help='the number of classes (default:2)')
    parser.add_argument('--num-proc', type=int, default=4,
                        help='the number of processes (default:4)')
    parser.add_argument('--selected-file', type=str,
                        default='./inference/cld_clr_tile_list.json',
                        help='list of files needed to compute boundary accuracy')
    parser.add_argument('--out-file', type=str,
                        default=None,
                        help='output file')

    args = parser.parse_args()

    net_root = {
        'DeeplabV3Plus-seed1': './inference/DeeplabV3Plus-seed1',
        'DeeplabV3Plus-seed2': './inference/DeeplabV3Plus-seed2',
        'DeeplabV3Plus-seed3': './inference/DeeplabV3Plus-seed3',
        'DeeplabV3Plus-seed4': './inference/DeeplabV3Plus-seed4',
    }

    for net in net_root.keys():
        args.pre_root = net_root[net]
        args.out_file = os.path.join('./inference-mix', net + '.json')
        print('prediction: {}'.format(args.pre_root))
        print('ground truth: {}'.format(args.gt_root))
        start = time.time()
        comparator = Comparator(args)
        comparator.cal_metric()
        comparator.p.close()  # 关闭进程池
        print('Using {}s!'.format(time.time() - start))
