""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class FMM(nn.Module):
    """Feature map module"""

    def __init__(self, in_channels):
        super(FMM, self).__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            DoubleConv(64, 128, 96)
        )
        self.stage2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(128, 256, 192)
        )
        self.stage3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(256, 512, 256)
        )

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)

        return x1, x2, x3


class Multiscale(nn.Module):
    """Multiscale module"""
    
    def __init__(self):
        super(Multiscale, self).__init__()

        self.scale1 = ScaleBlock(16)
        self.scale2 = ScaleBlock(8)
        self.scale3 = ScaleBlock(4)
        self.scale4 = ScaleBlock(2)

    def forward(self, x):
        x1 = self.scale1(x)
        x2 = self.scale2(x)
        x3 = self.scale3(x)
        x4 = self.scale4(x)

        # resolving image size inconsistencies
        maxH = max([x1.size()[2], x2.size()[2], x3.size()[2], x4.size()[2]])
        maxW = max([x1.size()[3], x2.size()[3], x3.size()[3], x4.size()[3]])
        x1 = self._padding(x1, maxH, maxW)
        x2 = self._padding(x2, maxH, maxW)
        x3 = self._padding(x3, maxH, maxW)
        x4 = self._padding(x4, maxH, maxW)
        return torch.cat((x1, x2, x3, x4), dim=1)

    def _padding(self, x, maxH, maxW):
        diffH, diffW = maxH - x.size()[2], maxW - x.size()[3]
        return F.pad(x, [diffW // 2, diffW - diffW // 2, diffH // 2, diffH - diffH // 2])


class ScaleBlock(nn.Module):
    """Used in multiscale module"""

    def __init__(self, pool_size):
        super(ScaleBlock, self).__init__()
        self.scale = nn.Sequential(
            nn.AvgPool2d(pool_size),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=pool_size, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.scale(x)


class Up(nn.Module):
    """convolution then upscaling"""

    def __init__(self, in_channels, out_channels, bn=False):
        super().__init__()
        if bn:
            self.conv_up = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            )
        else:
            self.conv_up = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            )

    def forward(self, x):
        return self.conv_up(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
