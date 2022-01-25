import torch
import argparse


def get_config_tr(net_name):
    # ---------------------------- #
    # create ArgumentParser object
    # ---------------------------- #
    parser = argparse.ArgumentParser(description="PyTorch {} Training".format(net_name))

    # ----------------------- #
    # add network information
    # ----------------------- #
    parser.add_argument('--net', type=str, default='{}'.format(net_name),
                        choices=['DeeplabV3Plus', 'MFCNN', 'MSCFF', 'MUNet',
                                 'TLNet', 'UNet', 'UNet-3', 'UNet-2', 'UNet-1',
                                 'UNet-dilation', 'UNetS3', 'UNetS2', 'UNetS1'],
                        help='network name (default: ?)')
    parser.add_argument('--in-channels', type=int, default=8,
                        help='number of input channels')
    # DeeplabV3Plus
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='use pretrained model in backbone (resnet101)')
    # MFCNN, RSNet
    parser.add_argument('--dropout-p', type=float, default=0.2,
                        help='probability of dropout layer')
    # UNet-dilation
    parser.add_argument('--dilation', type=int, default=2,
                        help='the rate of dilation convolution in UNet')

    # ----------------------- #
    # cpu, cuda, gpu and seed
    # ----------------------- #
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--num-proc', type=int, default=4,
                        metavar='N', help='metrics evaluation threads')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')  # compute unified device architecture
    parser.add_argument('--gpu-ids', type=str, default='0,1',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=3, metavar='S',  # the same seed create the same random number
                        help='random seed (default: 1)')  # to reduce the randomness of the results

    # --------------------- #
    # training hyper params
    # --------------------- #
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 600)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=24,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')

    # ---------------- #
    # optimizer params
    # ---------------- #
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal', 'wb'],
                        help='loss func type (default: ce)')
    parser.add_argument('--loss-interval', type=int, default=256,
                        help='print loss interval')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='step',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')  # optimal method to speed up convergence
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')  # reduce over-fitting
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')  # optimal method

    # -------------- #
    # checking point
    # -------------- #
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    parser.add_argument('--save-epoch', action='store_true', default=True,
                        help='save checkpoint every epoch')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')  # finetuning on a different dataset

    # ----------------- #
    # evaluation option
    # ----------------- #
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    # ---- #
    # data
    # ---- #
    parser.add_argument('--dataset', type=str, default='RS',
                        choices=['RS'], help='dataset name (default: RS)')
    parser.add_argument('--num-classes', type=int, default=2,
                        help='the number of classes (default:2)')
    parser.add_argument('--h', type=int, default=256,
                        help='image height')
    parser.add_argument('--w', type=int, default=256,
                        help='image width')
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
    parser.add_argument('--mean', type=str,
                        default='0.432, 0.398, 0.411, 0.479, 0.240, 0.192, 0.037, 268.051',
                        help='mean of each channel (used in data normalization), \
                        must be a comma-separated list of floats only \
                        (default: 0.432, 0.398, 0.411, 0.479, 0.240, 0.192, 0.037, 268.051)')
    parser.add_argument('--std', type=str,
                        default='0.313, 0.295, 0.311, 0.285, 0.162, 0.132, 0.079, 25.412',
                        help='standard deviation of each channel (used in data normalization), \
                            must be a comma-separated list of floats only \
                            (default: 0.313, 0.295, 0.311, 0.285, 0.162, 0.132, 0.079, 25.412)')

    args = parser.parse_args()  # analyze parameters
    args.cuda = not args.no_cuda and torch.cuda.is_available()  # CUDA can work
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:  # multiple gpu
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size, lr and data info
    if args.epochs is None:
        epoches = {
            'rs': 100
        }
        args.epochs = epoches[args.dataset.lower()]  # .lower() convert uppercase to lowercase

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)  # relate to the time and accuracy of gradient descent

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'rs': 0.1,
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size  # learning rate

    args.mean = [float(s) for s in args.mean.split(',')][0:args.in_channels]
    args.std = [float(s) for s in args.std.split(',')][0:args.in_channels]
    # args.in_channels = len(args.mean)

    if args.checkname is None:
        if args.net == 'DeeplabV3Plus':
            args.checkname = 'deeplab-' + str(args.backbone) + \
                             '-outputstride' + str(args.out_stride) + \
                             '-in_channels' + str(args.in_channels) + \
                             '-lr' + str(args.lr) + \
                             '-batch' + str(args.batch_size) + \
                             '-seed' + str(args.seed)
        elif args.net == 'UNet-dilation':
            args.checkname = args.net + \
                             '-UNetD' + str(args.dilation) + \
                             '-in_channels' + str(args.in_channels) + \
                             '-lr' + str(args.lr) + \
                             '-batch' + str(args.batch_size) + \
                             '-seed' + str(args.seed)
        else:
            args.checkname = args.net + \
                             '-in_channels' + str(args.in_channels) + \
                             '-lr' + str(args.lr) + \
                             '-batch' + str(args.batch_size) + \
                             '-seed' + str(args.seed)

    return args


def get_config_test(net_name):
    # ---------------------------- #
    # create ArgumentParser object
    # ---------------------------- #
    parser = argparse.ArgumentParser(description="PyTorch {} Inference".format(net_name))

    # ----------------------- #
    # add network information
    # ----------------------- #
    parser.add_argument('--net', type=str, default='{}'.format(net_name),
                        choices=['DeeplabV3Plus', 'MFCNN', 'MSCFF', 'MUNet',
                                 'TLNet', 'UNet', 'UNet-3', 'UNet-2', 'UNet-1',
                                 'UNet-dilation', 'UNetS3', 'UNetS2', 'UNetS1'],
                        help='network name (default: ?)')
    parser.add_argument('--in-channels', type=int, default=8,
                        help='number of input channels')
    # DeeplabV3Plus
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=8,
                        help='network output stride (default: 8)')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='use pretrained model in backbone (resnet101)')
    # MFCNN, RSNet
    parser.add_argument('--dropout-p', type=float, default=0.2,
                        help='probability of dropout layer')
    # UNet-dilation
    parser.add_argument('--dilation', type=int, default=1,
                        help='the rate of dilation convolution in UNet')

    # ----------------------- #
    # cpu, cuda, gpu and seed
    # ----------------------- #
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--num-proc', type=int, default=2,
                        metavar='N', help='metrics evaluation threads')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')  # compute unified device architecture
    parser.add_argument('--gpu-ids', type=str, default='0,1',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',  # the same seed create the same random number
                        help='random seed (default: 1)')  # to reduce the randomness of the results

    # --------------------- #
    # inference hyper params
    # --------------------- #
    parser.add_argument('--batch-size', type=int, default=64,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal', 'wb'],
                        help='loss func type (default: ce)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')

    # -------------- #
    # checking point
    # -------------- #
    parser.add_argument('--load-paths', type=str, default=None,
                        help='put the model path, must be a comma-separated list')

    # ---- #
    # data
    # ---- #
    parser.add_argument('--dataset', type=str, default='RS',
                        choices=['RS'], help='dataset name (default: RS)')
    parser.add_argument('--num-classes', type=int, default=2,
                        help='the number of classes (default:2)')
    parser.add_argument('--h', type=int, default=256,
                        help='image height')
    parser.add_argument('--w', type=int, default=256,
                        help='image width')
    parser.add_argument('--test-root', type=str,
                        default='./example/test/Images',
                        help='image root of test set')
    parser.add_argument('--test-list', type=str,
                        default='./example/test/test.txt',
                        help='image list of test set')
    parser.add_argument('--no-gt', action='store_true', default=False,
                        help='no available ground truth')
    parser.add_argument('--mean', type=str,
                        default='0.432, 0.398, 0.411, 0.479, 0.240, 0.192, 0.037, 268.051',
                        help='mean of each channel (used in data normalization), \
                        must be a comma-separated list of floats only \
                        (default: 0.432, 0.398, 0.411, 0.479, 0.240, 0.192, 0.037, 268.051)')
    parser.add_argument('--std', type=str,
                        default='0.313, 0.295, 0.311, 0.285, 0.162, 0.132, 0.079, 25.412',
                        help='standard deviation of each channel (used in data normalization), \
                            must be a comma-separated list of floats only \
                            (default: 0.313, 0.295, 0.311, 0.285, 0.162, 0.132, 0.079, 25.412)')

    # ------ #
    # output
    # ------ #
    parser.add_argument('--out-path', type=str, default=None,
                        help='result output path')
    parser.add_argument('--save-img', action='store_true', default=True,
                        help='save predicted image. If no_gt is true, \
                        the option fails, i.e. always true')

    args = parser.parse_args()  # analyze parameters
    args.cuda = not args.no_cuda and torch.cuda.is_available()  # CUDA can work
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:  # multiple gpu
            args.sync_bn = True
        else:
            args.sync_bn = False

    if args.load_paths:
        args.load_paths = args.load_paths.split(sep=',')

    # default settings for batch_size and data info
    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)  # relate to the time and accuracy of gradient descent

    args.mean = [float(s) for s in args.mean.split(',')][0:args.in_channels]
    args.std = [float(s) for s in args.std.split(',')][0:args.in_channels]
    # args.in_channels = len(args.mean)

    return args


def get_config_erf(net_name):
    # ---------------------------- #
    # create ArgumentParser object
    # ---------------------------- #
    parser = argparse.ArgumentParser(description="PyTorch {} (Calculate ERF)".format(net_name))

    # ----------------------- #
    # add network information
    # ----------------------- #
    parser.add_argument('--net', type=str, default='{}'.format(net_name),
                        choices=['DeeplabV3Plus', 'MFCNN', 'MSCFF', 'MUNet',
                                 'TLNet', 'UNet', 'UNet-3', 'UNet-2', 'UNet-1',
                                 'UNet-dilation', 'UNetS3', 'UNetS2', 'UNetS1'],
                        help='network name (default: ?)')
    parser.add_argument('--in-channels', type=int, default=8,
                        help='number of input channels')
    # DeeplabV3Plus
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=8,
                        help='network output stride (default: 8)')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='use pretrained model in backbone (resnet101)')
    # MFCNN, RSNet
    parser.add_argument('--dropout-p', type=float, default=0.2,
                        help='probability of dropout layer')
    # UNet-dilation
    parser.add_argument('--dilation', type=int, default=1,
                        help='the rate of dilation convolution in UNet')

    # ----------------------- #
    # cpu, cuda, gpu and seed
    # ----------------------- #
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')  # compute unified device architecture
    parser.add_argument('--gpu-ids', type=str, default='0,1',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',  # the same seed create the same random number
                        help='random seed (default: 1)')  # to reduce the randomness of the results

    # --------------------- #
    # erf hyper params
    # --------------------- #
    parser.add_argument('--batch-size', type=int, default=32,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--max-pro', action='store_true', default=False,
                        help='whether to use the max class probability')

    # -------------- #
    # checking point
    # -------------- #
    parser.add_argument('--load-paths', type=str, default=None,
                        help='put the model path, must be a comma-separated list')

    # ---- #
    # data
    # ---- #
    parser.add_argument('--dataset', type=str, default='RS',
                        choices=['RS'], help='dataset name (default: RS)')
    parser.add_argument('--num-classes', type=int, default=2,
                        help='the number of classes (default:2)')
    parser.add_argument('--img-root', type=str,
                        default='./example/test/Images',
                        help='image root of train set')
    parser.add_argument('--pixel-list', type=str,
                        default='./erf/selected_pixel_all.json',
                        help='image list of train set')
    parser.add_argument('--mean', type=str,
                        default='0.432, 0.398, 0.411, 0.479, 0.240, 0.192, 0.037, 268.051',
                        help='mean of each channel (used in data normalization), \
                        must be a comma-separated list of floats only \
                        (default: 0.432, 0.398, 0.411, 0.479, 0.240, 0.192, 0.037, 268.051)')
    parser.add_argument('--std', type=str,
                        default='0.313, 0.295, 0.311, 0.285, 0.162, 0.132, 0.079, 25.412',
                        help='standard deviation of each channel (used in data normalization), \
                            must be a comma-separated list of floats only \
                            (default: 0.313, 0.295, 0.311, 0.285, 0.162, 0.132, 0.079, 25.412)')

    # ------ #
    # output
    # ------ #
    parser.add_argument('--out-path', type=str, default=None,
                        help='result output path')
    parser.add_argument('--save-img', action='store_true', default=False,
                        help='save predicted image. If no_gt is true, \
                        the option fails, i.e. always true')

    args = parser.parse_args()  # analyze parameters
    args.cuda = not args.no_cuda and torch.cuda.is_available()  # CUDA can work
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:  # multiple gpu
            args.sync_bn = True
        else:
            args.sync_bn = False

    if args.load_paths:
        args.load_paths = args.load_paths.split(sep=',')

    # default settings for batch_size and data info
    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)  # relate to the time and accuracy of gradient descent

    args.mean = [float(s) for s in args.mean.split(',')][0:args.in_channels]
    args.std = [float(s) for s in args.std.split(',')][0:args.in_channels]
    # args.in_channels = len(args.mean)

    return args
