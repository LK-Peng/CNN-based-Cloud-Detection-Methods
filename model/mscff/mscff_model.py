""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .mscff_parts import *


class MSCFF(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(MSCFF, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # encoder
        self.inc = CBRR(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.encoder_DCBRR1 = CBRR(512, 512, 2)
        self.encoder_DCBRR2 = CBRR(512, 512, 4)

        # decoder
        self.decoder_DCBRR1 = CBRR(512, 512, 4)
        self.decoder_DCBRR2 = CBRR(512, 512, 2)
        self.decoder_CBRR = CBRR(512, 512)
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)

        # multi-scale feature fusion
        self.ms_block1 = MultiScaleBlock(512, n_classes, scale_factor=8)
        self.ms_block2 = MultiScaleBlock(512, n_classes, scale_factor=8)
        self.ms_block3 = MultiScaleBlock(512, n_classes, scale_factor=8)
        self.ms_block4 = MultiScaleBlock(256, n_classes, scale_factor=4)
        self.ms_block5 = MultiScaleBlock(128, n_classes, scale_factor=2)
        self.ms_block6 = MultiScaleBlock(64, n_classes)

        # out
        self.outc = OutConv(6 * n_classes, n_classes)

    def forward(self, x):
        # encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.encoder_DCBRR1(x4)
        x6 = self.encoder_DCBRR2(x5)

        # decoder
        x7 = self.decoder_DCBRR1(x6) + x6
        x8 = self.decoder_DCBRR2(x7) + x5
        x9 = self.decoder_CBRR(x8) + x4

        # resolving image size inconsistencies
        x10 = self.up1(x9)
        diffH, diffW = x3.size()[2] - x10.size()[2], x3.size()[3] - x10.size()[3]
        x10 = F.pad(x10, [diffW // 2, diffW - diffW // 2, diffH // 2, diffH - diffH // 2]) + x3

        # resolving image size inconsistencies
        x11 = self.up2(x10)
        diffH, diffW = x2.size()[2] - x11.size()[2], x2.size()[3] - x11.size()[3]
        x11 = F.pad(x11, [diffW // 2, diffW - diffW // 2, diffH // 2, diffH - diffH // 2]) + x2

        # resolving image size inconsistencies
        x12 = self.up3(x11)
        diffH, diffW = x1.size()[2] - x12.size()[2], x1.size()[3] - x12.size()[3]
        x12 = F.pad(x12, [diffW // 2, diffW - diffW // 2, diffH // 2, diffH - diffH // 2]) + x1

        # multi-scale feature fusion
        x7, x8, x9, x10, x11, x12 = self._padding(self.ms_block1(x7), x.size()[2], x.size()[3]), \
                                    self._padding(self.ms_block2(x8), x.size()[2], x.size()[3]), \
                                    self._padding(self.ms_block3(x9), x.size()[2], x.size()[3]), \
                                    self._padding(self.ms_block4(x10), x.size()[2], x.size()[3]), \
                                    self._padding(self.ms_block5(x11), x.size()[2], x.size()[3]), \
                                    self._padding(self.ms_block6(x12), x.size()[2], x.size()[3])
        x_ms = torch.cat((x7, x8, x9, x10, x11, x12), dim=1)

        logits = self.outc(x_ms)
        return logits

    def _padding(self, x, maxH, maxW):
        diffH, diffW = maxH - x.size()[2], maxW - x.size()[3]
        return F.pad(x, [diffW // 2, diffW - diffW // 2, diffH // 2, diffH - diffH // 2])
