#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''

Author   : MG Studio
Datetime : 2018/11/8
Filename : U_net2.py
'''

import numpy as np
import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable

# 将两个卷积层简单封装一下
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), #Conv层
            nn.BatchNorm2d(out_ch), #BN层
            nn.ReLU(inplace=True), #ReLU层
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

        )

    def forward(self, input):
        return self.conv(input)

class Unet2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet2, self).__init__()
        self.conv1 = DoubleConv(in_ch, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5= DoubleConv(256, 512)

        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6 = DoubleConv(512,256)
        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7 = DoubleConv(256, 128)
        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8 = DoubleConv(128, 64)
        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9 = DoubleConv(64, 32)

        self.conv10 = nn.Conv2d(32, out_ch, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)

        u6 = self.up6(c5)
        m6 = torch.cat([u6, c4], dim=1)

        c6 = self.conv6(m6)
        u7 = self.up7(c6)
        m7 = torch.cat([u7, c3], dim=1)
        c7 = self.conv7(m7)
        u8 = self.up8(c7)
        m8 = torch.cat([u8, c2], dim=1)
        c8 = self.conv8(m8)
        u9 = self.up9(c8)
        m9 = torch.cat([u9, c1], dim=1)
        c9 = self.conv9(m9)
        c10 = self.conv10(c9)

        out = nn.Sigmoid()(c10)
        return out

if __name__ == '__main__':

    from Load_Data import *
    # 实例化一个训练集对象
    train_dataset = Train_Data(TRAIN_DATASETS_ROOT, train=True)
    Unet = Unet2(3, 1)
    img, msk = train_dataset[0]
    img = Variable(img.unsqueeze(0))
    print(img.shape)
    out = Unet(img)
    print(out.shape)

    # 模型可视化
    import tensorboardX

    # 定义一个tensorboardX的写对象直接画模型
    with tensorboardX.SummaryWriter("./logs/") as writer:
        writer.add_graph(Unet, (img))
