# CNN-based-Cloud-Detection-Methods
## Paper: Understanding the Role of Receptive Field of Convolutional Neural Network for Cloud Detection in Landsat 8 OLI Imagery

### TODO
- [x] Support different convolutional neural networks for cloud detection
- [x] Support calculation of effective receptive field
- [x] Multi-GPU training



The links of the trained models are as follows:
|Input Band Number|Download Link|Password|
|:-:|:-:|:-:|
|8|[Baidu Netdisk](https://pan.baidu.com/s/1obbeQlKybN40EW5lO6XUqQ?pwd=3tre)|3tre|
|6|[Baidu Netdisk](https://pan.baidu.com/s/1xAf6PnOfokroxmcQlIUhUA?pwd=m6nt)|m6nt|
|4|[Baidu Netdisk](https://pan.baidu.com/s/1nYHaIWZ0aA3MsxqHdviG5Q?pwd=qy48)|qy48|

The trained model for the input data of 8 channels can also be downloaded from **[Google Drive](https://drive.google.com/drive/folders/1Av1Gl3WEug_G2UC4WZgddI1YVdvCiwfW?usp=sharing)**



### Introduction
This is a PyTorch(1.7.1) implementation of varied convolutional neural networks (CNNs) for cloud detection in Landsat 8 OLI imagery. Currently, we train these networks using [L8 Biome](https://landsat.usgs.gov/landsat-8-cloud-cover-assessment-validation-data) dataset. The related paper aims to understand the role of receptive field of CNN for cloud detection in Landsat 8 OLI imagery and is under review.



### Installation
The code was tested with **Anaconda** and **Python 3.7.3**.

0. For **PyTorch** dependency, see [pytorch.org](https://pytorch.org/) for more details.

1. For **Captum** dependency used for computing the effective receptive field, see [captum.ai](https://captum.ai/) for more details.

2. For **GDAL** dependency used for reading and writing raster data, use version 2.3.3.



### Training
Follow steps below to train your model

0. Configure your dataset path in [config.py](https://github.com/LK-Peng/CNN-based-Cloud-Detection-Methods/blob/main/config.py)
    ```Shell
    def get_config_tr(net_name):
      ...
      parser.add_argument('--train-root', type=str,
                          default='./example/train/Images',
                          help='image root of train set')
      parser.add_argument('--train-list', type=str,
                          default='./example/train/train.txt',
                          help='image list of train set')
      parser.add_argument('--val-root', type=str,
                          default='./example/val/Images',
                          help='image root of validation set')
      parser.add_argument('--val-list', type=str,
                          default='./example/val/val.txt',
                          help='image list of validation set')
    ```
    
1. Configure the network you want to use in [config.py](https://github.com/LK-Peng/CNN-based-Cloud-Detection-Methods/blob/main/config.py) 
    ```Shell
    def get_config_tr(net_name):
      ...
      parser.add_argument('--net', type=str, default='{}'.format(net_name),
                          choices=['DeeplabV3Plus', 'MFCNN', 'MSCFF', 'MUNet',
                                   'TLNet', 'UNet', 'UNet-3', 'UNet-2', 'UNet-1',
                                   'UNet-dilation', 'UNetS3', 'UNetS2', 'UNetS1'],
                          help='network name (default: ?)')
    ```
    
    or [train.py](https://github.com/LK-Peng/CNN-based-Cloud-Detection-Methods/blob/main/train.py)
    
    ```Shell
     def main():
       # choices=['DeeplabV3Plus', 'MFCNN', 'MSCFF', 'MUNet', 'TLNet', 'UNet', 'UNet-3', 'UNet-2', 'UNet-1', 'UNet-dilation', 'UNetS3', 'UNetS2', 'UNetS1']
       args = get_config_tr('TLNet')
     ```

2. Run script
     ```Shell
     python train.py
     ```
     
### Others
0. [inference.py](https://github.com/LK-Peng/CNN-based-Cloud-Detection-Methods/blob/main/inference.py) is used for predicting cloud detection results and output accuracies.

1. [erf.py](https://github.com/LK-Peng/CNN-based-Cloud-Detection-Methods/blob/main/erf.py) is used for computing the effective receptive field

2. [comparator.py](https://github.com/LK-Peng/CNN-based-Cloud-Detection-Methods/blob/main/comparator.py) is used for computing the accuracies of the predicted results.



### Acknowledgement
[DeepLab-V3-Plus](https://github.com/jfzhang95/pytorch-deeplab-xception)
[UNet](https://github.com/milesial/Pytorch-UNet)
