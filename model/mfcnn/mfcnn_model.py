""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .mfcnn_parts import *


class MFCNN(nn.Module):
    def __init__(self, n_channels, n_classes, dropout_p=0.2):
        super(MFCNN, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # feature map module
        self.fmm = FMM(n_channels)

        # multiscale module
        self.msm = Multiscale()

        # up-sampling module
        self.up1 = Up(1536, 512)
        self.up2 = Up(768, 256, bn=True)
        self.up3 = Up(384, 128, bn=True)

        # out
        self.dp = nn.Dropout(p=dropout_p)
        self.outc = OutConv(128, n_classes)

    def forward(self, x):
        # feature map module
        x1, x2, x3 = self.fmm(x)

        # multiscale module
        x4 = self.msm(x3)

        # up-sampling module
        # resolving image size inconsistencies
        diffH, diffW = x3.size()[2] - x4.size()[2], x3.size()[3] - x4.size()[3]
        x4 = F.pad(x4, [diffW // 2, diffW - diffW // 2, diffH // 2, diffH - diffH // 2])
        x = self.up1(torch.cat((x3, x4), dim=1))

        # resolving image size inconsistencies
        diffH, diffW = x2.size()[2] - x.size()[2], x2.size()[3] - x.size()[3]
        x = F.pad(x, [diffW // 2, diffW - diffW // 2, diffH // 2, diffH - diffH // 2])
        x = self.up2(torch.cat((x2, x), dim=1))

        # resolving image size inconsistencies
        diffH, diffW = x1.size()[2] - x.size()[2], x1.size()[3] - x.size()[3]
        x = F.pad(x, [diffW // 2, diffW - diffW // 2, diffH // 2, diffH - diffH // 2])
        x = self.up3(torch.cat((x1, x), dim=1))

        # out
        x = self.dp(x)
        return self.outc(x)
