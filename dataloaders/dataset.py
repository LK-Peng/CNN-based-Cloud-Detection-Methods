import os
import json
import numpy as np
from osgeo import gdal
from torchvision import transforms
from torch.utils.data import Dataset

import dataloaders.custom_transforms as ctr
# import custom_transforms as ctr


class RSSet(Dataset):
    def __init__(self, args, split):
        super().__init__()
        if split == 'train':
            set_list = args.train_list
            set_root = args.train_root
        elif split == 'val':
            set_list = args.val_list
            set_root = args.val_root
        elif split == 'test':
            set_list = args.test_list
            set_root = args.test_root
        with open(set_list, 'r') as f:
            lines = f.readlines()
        img_files = []
        mask_files = []
        for line in lines:
            img_files.append(os.path.join(set_root, line.split()[0]))
            mask_files.append(img_files[-1].replace('Images', 'Masks'))
        self.img_files = img_files
        self.mask_files = mask_files
        self.split = split
        self.args = args

    def __getitem__(self, item):
        img = gdal.Open(self.img_files[item]).ReadAsArray()[0:self.args.in_channels, :]
        if self.split == 'train':
            sample = {'image': img,
                      'label': self._transform_mask_multi(gdal.Open(self.mask_files[item]).ReadAsArray())}
            return self._transform_tr(sample)
        elif self.split == 'val':
            sample = {'image': img,
                      'label': self._transform_mask_multi(gdal.Open(self.mask_files[item]).ReadAsArray())}
            return self._transform_tr(sample)
        elif self.split == 'test':
            sample = {'image': img, 'label': np.array([])}
            return self._transform_test(sample)

    def _transform_tr(self, img):
        data_transforms = transforms.Compose([
            ctr.Normalize(mean=self.args.mean, std=self.args.std),
            ctr.ToTensor()
        ])
        return data_transforms(img)

    def _transform_val(self, img):
        data_transforms = transforms.Compose([
            ctr.Normalize(mean=self.args.mean, std=self.args.std),
            ctr.ToTensor()
        ])
        return data_transforms(img)

    def _transform_test(self, img):
        data_transforms = transforms.Compose([
            ctr.Normalize(mean=self.args.mean, std=self.args.std, no_gt=True),
            ctr.ToTensor(no_gt=True)
        ])
        return data_transforms(img)

    def _transform_mask_binary(self, mask):
        # convert mask value to 0,1,2...
        # mask(mix2): 1 -- clear, 2 -- cloud
        mask[mask == 1] = 0
        mask[mask == 2] = 1

        return mask

    def _transform_mask_multi(self, mask):
        # convert mask value to 0,1,2...
        # mask(mix): 0 -- fill, 64 -- cloud shadow, 128 -- clear, 192 -- thin cloud, 255 -- cloud
        mask[mask == 64] = 0
        mask[mask == 128] = 0
        mask[mask == 192] = 1
        mask[mask == 255] = 1

        return mask

    def __len__(self):
        return len(self.img_files)


class RSERFSet(Dataset):
    def __init__(self, args):
        super().__init__()

        """
        Example: 
            pixels = {'Barren_02_0319.txt': [[121, 122], [125,127], [122,131]]}
        """
        with open(args.pixel_list, 'r') as f:
            pixels = json.load(f)
        selected_files = list(pixels.keys())
        img_files, targets = [], []
        for key in pixels.keys():
            if key in selected_files:
                img_files.extend([os.path.join(args.img_root, key)] * len(pixels[key]))
                targets.extend(pixels[key])
        self.img_files = img_files
        self.targets = targets
        self.args = args

    def __getitem__(self, item):
        img = gdal.Open(self.img_files[item]).ReadAsArray()[0:self.args.in_channels, :]
        sample = {'image': img,
                  'label': np.array(self.targets[item])}
        return self._transform_erf(sample)

    def _transform_erf(self, img):
        data_transforms = transforms.Compose([
            ctr.Normalize(mean=self.args.mean, std=self.args.std, no_gt=True),
            ctr.ToTensor(no_gt=True)
        ])
        return data_transforms(img)

    def __len__(self):
        return len(self.img_files)


class MaskSet(Dataset):
    def __init__(self, args):
        super().__init__()

        filelist = os.listdir(args.pre_root)
        with open('/home/clouddt/XAI/dataFinal/inference/cld_clr_tile_list.json', 'r') as f:
            selected_files = json.load(f)
        self.pre_files = [os.path.join(args.pre_root, file) for file in filelist if file in selected_files]
        self.gt_files = [os.path.join(args.gt_root, os.path.split(file)[-1]) for file in self.pre_files]
        # in real ground truth mask:
        # mask(mix): 0 -- fill, 64 -- cloud shadow, 128 -- clear, 192 -- thin cloud, 255 -- cloud
        self.merge_class = args.merge_class

    def __getitem__(self, item):
        sample = {'pre': gdal.Open(self.pre_files[item]).ReadAsArray(),
                  'gt': gdal.Open(self.gt_files[item]).ReadAsArray()}
        if self.merge_class:
            sample['gt'] = self._transform_mask_multi(sample['gt'])
        return sample

    def _transform_mask_multi(self, mask):
        # convert mask value to 0,1,2...
        # mask(mix): 0 -- fill, 64 -- cloud shadow, 128 -- clear, 192 -- thin cloud, 255 -- cloud
        mask[mask == 64] = 0
        mask[mask == 128] = 0
        mask[mask == 192] = 1
        mask[mask == 255] = 1

        return mask

    def __len__(self):
        return len(self.pre_files)
        

if __name__ == "__main__":
    import os
    import argparse
    import numpy as np
    from osgeo import gdal
    from tqdm import tqdm
    from torch.utils.data import DataLoader


    def save_img(tiff, out_file, projection=None, geotransform=None):
        # save tiff image

        NP2GDAL_CONVERSION = {
            "uint8": 1,
            "int8": 1,
            "uint16": 2,
            "int16": 3,
            "uint32": 4,
            "int32": 5,
            "float32": 6,
            "float64": 7,
            "complex64": 10,
            "complex128": 11,
        }  # convert np to gdal
        gdal_type = NP2GDAL_CONVERSION[tiff.dtype.name]
        if len(tiff.shape) == 2:
            tiff = np.expand_dims(tiff, axis=0)
        channel, row, col = tiff.shape
        # 使用驱动对象来创建数据集
        gtiff_driver = gdal.GetDriverByName('GTiff')
        out_ds = gtiff_driver.Create(out_file, col, row, channel, gdal_type)
        if projection is not None and geotransform is not None:
            out_ds.SetProjection(projection)  # 设置投影
            out_ds.SetGeoTransform(geotransform)  # 设置geotransform信息
        # 向输出数据源写入数据
        for iband in range(channel):
            out_ds.GetRasterBand(iband + 1).WriteArray(tiff[iband, :, :])
        del out_ds

    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")  # create ArgumentParser object
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--batch-size', type=int, default=16,
                        metavar='N', help='input batch size for \
                                    training (default: auto)')
    parser.add_argument('--train-root', type=str, default='./example/train/Images',
                        help='image root of train set')
    parser.add_argument('--train-list', type=str, default='./example/train/train.txt',
                        help='image list of train set')
    parser.add_argument('--val-root', type=str, default='./example/val/Images',
                        help='image root of validation set')
    parser.add_argument('--val-list', type=str, default='./example/val/val.txt',
                        help='image list of validation set')
    parser.add_argument('--mean', type=str,
                        default='0.432, 0.398, 0.411, 0.479, 0.240, 0.192, 0.037, 268.051',
                        help='mean of each channel (used in data normalization), \
                            must be a comma-separated list of floats only')
    parser.add_argument('--std', type=str,
                        default='0.313, 0.295, 0.311, 0.285, 0.162, 0.132, 0.079, 25.412',
                        help='standard deviation of each channel (used in data normalization), \
                                must be a comma-separated list of floats only')
    parser.add_argument('--output-root', type=str, default='./example/train/check',
                        help='image root of train set')

    args = parser.parse_args()  # analyze parameters

    args.mean = [float(s) for s in args.mean.split(',')]
    args.std = [float(s) for s in args.std.split(',')]

    kwargs = {'num_workers': args.workers, 'pin_memory': True}
    train_set = RSSet(args, split='train')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    tbar = tqdm(train_loader, desc='\r')
    for i, sample in enumerate(tbar):
        if i == 0:
            image, target = sample['image'].numpy(), sample['label'].numpy()
            for j in range(args.batch_size):
                # save image
                image_temp = image[j, :]
                save_img(image_temp, os.path.join(args.output_root, 'train_check_{}.tif'.format(j)))
                # save target
                target_temp = target[j, :]
                save_img(target_temp, os.path.join(args.output_root, 'train_check_target_{}.tif'.format(j)))
            break

    val_set = RSSet(args, split='val')
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    tbar = tqdm(val_loader, desc='\r')
    for i, sample in enumerate(tbar):
        if i == 0:
            image, target = sample['image'].numpy(), sample['label'].numpy()
            for j in range(args.batch_size):
                # save image
                image_temp = image[j, :]
                save_img(image_temp, os.path.join(args.output_root, 'val_check_{}.tif'.format(j)))
                # save target
                target_temp = target[j, :]
                save_img(target_temp, os.path.join(args.output_root, 'val_check_target_{}.tif'.format(j)))
            break

