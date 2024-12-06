""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, head_classes, vac_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.head_classes = head_classes
        self.vac_classes = vac_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.out_head = OutConv_new(64, self.head_classes, 32)
        self.out_vac = OutConv_new(64, self.vac_classes, 32)
        self.logsigma = nn.Parameter(torch.FloatTensor([-0.5, -0.5]))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        head = self.out_head(x)
        vac = self.out_vac(x)
        return head, vac#, self.logsigma

class UNet_sharp(nn.Module):
    def __init__(self, n_channels, head_classes, vac_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.head_classes = head_classes
        self.vac_classes = vac_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.out_head = OutConv_new(64, self.head_classes, 32)
        self.out_vac = OutConv_new(64, self.vac_classes, 32)
        self.logsigma = nn.Parameter(torch.FloatTensor([-0.5, -0.5, -0.5]))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=512, out_features=41)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        head = self.out_head(x)
        vac = self.out_vac(x)
        cls_head = self.avgpool(x5)
        cls = self.fc(cls_head)
        return head, vac, self.logsigma, cls
