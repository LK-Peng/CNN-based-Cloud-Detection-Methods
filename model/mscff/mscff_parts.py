""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class CBRR(nn.Module):
    """Convolution/Dilated Convolution, BN, ReLU with residual unit"""

    def __init__(self, in_channels, out_channels, dilation=1):
        super(CBRR, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block1(x)
        return self.block2(x) + x


class Down(nn.Module):
    """Downscaling with maxpool then CBRR"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            CBRR(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then CBRR"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1),
            CBRR(in_channels, out_channels)
        )

    def forward(self, x):
        return self.up_conv(x)


class MultiScaleBlock(nn.Module):
    """Block in multi-scale feature fusion"""

    def __init__(self, in_channels, out_channels, scale_factor=None):
        super(MultiScaleBlock, self).__init__()

        if scale_factor:
            self.ms_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
            )
        else:
            self.ms_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.ms_block(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(x)
