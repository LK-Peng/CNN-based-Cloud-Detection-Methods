from .deeplab import DeepLab
from .mfcnn import MFCNN
from .mscff import MSCFF
from .msunet import MSUNet
from .tlunet import TLUNet
from .unet import UNet
from .unet_3 import UNet_3
from .unet_2 import UNet_2
from .unet_1 import UNet_1
from .unet_dilation import UNet_dil
from .unet_s3 import UNetS3
from .unet_s2 import UNetS2
from .unet_s1 import UNetS1


def get_network(args):
    if args.net == 'DeeplabV3Plus':
        net = DeepLab(backbone=args.backbone, output_stride=args.out_stride,
                      num_classes=args.num_classes, in_channels=args.in_channels,
                      sync_bn=args.sync_bn, freeze_bn=args.freeze_bn,
                      pretrained=args.pretrained)
    elif args.net == 'MFCNN':
        net = MFCNN(n_channels=args.in_channels, n_classes=args.num_classes, dropout_p=args.dropout_p)
    elif args.net == 'MSCFF':
        net = MSCFF(n_channels=args.in_channels, n_classes=args.num_classes)
    elif args.net == 'MSUNet':
        net = MSUNet(n_channels=args.in_channels, n_classes=args.num_classes)
    elif args.net == 'TLUNet':
        net = TLUNet(n_channels=args.in_channels, n_classes=args.num_classes)
    elif args.net == 'UNet':
        net = UNet(n_channels=args.in_channels, n_classes=args.num_classes, bilinear=False)
    elif args.net == 'UNetS3':
        net = UNetS3(n_channels=args.in_channels, n_classes=args.num_classes, bilinear=False)
    elif args.net == 'UNetS2':
        net = UNetS2(n_channels=args.in_channels, n_classes=args.num_classes, bilinear=False)
    elif args.net == 'UNetS1':
        net = UNetS1(n_channels=args.in_channels, n_classes=args.num_classes, bilinear=False)
    elif args.net == 'UNet-3':
        net = UNet_3(n_channels=args.in_channels, n_classes=args.num_classes, bilinear=False)
    elif args.net == 'UNet-2':
        net = UNet_2(n_channels=args.in_channels, n_classes=args.num_classes, bilinear=False)
    elif args.net == 'UNet-1':
        net = UNet_1(n_channels=args.in_channels, n_classes=args.num_classes, bilinear=False)
    elif args.net == 'UNet-dilation':
        net = UNet_dil(n_channels=args.in_channels, n_classes=args.num_classes, bilinear=False,
                       maxpool=False, dilation=args.dilation)
    else:
        raise NotImplementedError('The network {} is not supported yet'.format(args.net))

    return net


