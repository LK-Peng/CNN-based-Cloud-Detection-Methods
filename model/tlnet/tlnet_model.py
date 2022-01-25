""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .tlnet_parts import *


class TLNet(nn.Module):
    def __init__(self, n_channels, n_classes, dilations=[1, 1, 1, 1]):
        super(TLNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64, dilations=dilations[0:2])
        self.down2 = Down(64, 128, dilations=dilations[2:])
        self.up1 = Up(128, 64)
        self.up2 = Up(64, 32)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x = self.down2(x2)
        x = self.up1(x, x2)
        x = self.up2(x, x1)
        logits = self.outc(x)
        return logits
