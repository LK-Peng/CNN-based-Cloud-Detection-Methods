import torch
import torch.nn as nn


class NetConvert1(nn.Module):
    '''
    获取UNet第一个池化及之后的结构
    '''
    def __init__(self, model, net_name):
        super(NetConvert1, self).__init__()

        self.net_name = net_name
        # 去掉第一个池化层前的层
        self.down1 = model.down1
        if net_name == 'UNet-2':
            self.down2 = model.down2
            self.up3 = model.up3
        if net_name == 'UNet-3':
            self.down2 = model.down2
            self.down3 =model.down3
            self.up2 = model.up2
            self.up3 = model.up3
        if net_name in ['UNet', 'UNet-dilation']:
            self.down2 = model.down2
            self.down3 = model.down3
            self.down4 = model.down4
            self.up1 = model.up1
            self.up2 = model.up2
            self.up3 = model.up3
        self.up4 = model.up4
        self.outc = model.outc

    def forward(self, x):
        x1 = x
        x2 = self.down1(x1)
        if self.net_name == 'UNet-2':
            x3 = self.down2(x2)
            x = self.up3(x3, x2)
        if self.net_name == 'UNet-3':
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x = self.up2(x4, x3)
            x = self.up3(x, x2)
        if self.net_name in ['UNet', 'UNet-dilation']:
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
        if self.net_name == 'UNet-1':
            x = self.up4(x2, x1)
        else:
            x = self.up4(x, x1)
        x = self.outc(x)
        return x


class NetConvert2(nn.Module):
    '''
    获取UNet中最后一个上采样层之后的结构
    '''
    def __init__(self, model):
        super(NetConvert2, self).__init__()

        # 截取出最后一个反卷积之后的层
        self.up4_conv = nn.Sequential(*list(model.up4.children())[1:])
        self.outc = model.outc

    def forward(self, x):
        '''
        x1: 上采样得到
        x: skip-connection得到
        '''
        x = self.up4_conv(x)
        x = self.outc(x)
        return x


class NetConvert3(nn.Module):
    '''
    获取UNet中最后一个上采样层及之前的结构
    '''
    def __init__(self, model, net_name):
        super(NetConvert3, self).__init__()

        self.net_name = net_name
        # 截取出最后一个反卷积之后的层
        self.inc = model.inc
        self.down1 = model.down1
        if net_name == 'UNet-2':
            self.down2 = model.down2
            self.up3 = model.up3
        if net_name == 'UNet-3':
            self.down2 = model.down2
            self.down3 = model.down3
            self.up2 = model.up2
            self.up3 = model.up3
        if net_name in ['UNet', 'UNet-dilation']:
            self.down2 = model.down2
            self.down3 = model.down3
            self.down4 = model.down4
            self.up1 = model.up1
            self.up2 = model.up2
            self.up3 = model.up3
        self.up_conv = model.up4.up

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        if self.net_name == 'UNet-2':
            x3 = self.down2(x2)
            x = self.up3(x3, x2)
        if self.net_name == 'UNet-3':
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x = self.up2(x4, x3)
            x = self.up3(x, x2)
        if self.net_name in ['UNet', 'UNet-dilation']:
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
        if self.net_name == 'UNet-1':
            x = self.up_conv(x2)
        else:
            x = self.up_conv(x)
        return x


class NetConvertShort(nn.Module):
    '''
    获取UNet中最后一个上采样层之后的结构
    '''
    def __init__(self, model):
        super(NetConvertShort, self).__init__()

        self.inc = model.inc
        # 截取出最后一个反卷积之后的层
        self.up4_conv = nn.Sequential(*list(model.up4.children())[1:])
        self.outc = model.outc

    def forward(self, x):
        '''
        x1: 上采样得到
        x: skip-connection得到
        '''
        x1 = self.inc(x[:, 0:8, :])
        x1 = torch.cat([x1, x[:, 8:, :]], dim=1)
        x1 = self.up4_conv(x1)
        x1 = self.outc(x1)
        return x1
