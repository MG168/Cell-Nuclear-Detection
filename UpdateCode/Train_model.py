#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''

Author   : MG Studio
Datetime : 2018/11/8
Filename : Train_model.py
'''

import torch
from torch.utils.data import DataLoader
import os
from Load_Data import *
from U_net import *
from U_net2 import *
from viewloss import ViewLoss
from Visdom import *

# 记录loss
losses_hist = [[], []]
# 绘制训练曲线
labels = ['train_loss', 'val_loss']
# 实例化view
view = ViewLoss(labels=labels, losses_hist=losses_hist)

#--------------------------------------------
# 训练参数设置
#----------------------------------------------
os.environ['CUDA_VISIBLE_DEVICE']='0' # 设置使用gpu0
BATCH_SIZE = 6 # 批次大小
EPOCHS = 20 # 迭代轮数
save_model_name = 'unet2' # 保存训练模型的名字
# loss权重系数
alpha = 0.4

# 训练loss可视化
# vis = Visualizer(env='main', win='Train', title='Loss',plot_together=False)

#--------------------------------------------
# Dice_cross计算
#----------------------------------------------
def dice_cross(pred, target):
    smooth = 1. # 平滑项
    iflat = pred.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    dice_coef = (2. * intersection + smooth) / (iflat.sum()+ tflat.sum() + smooth)
    return 1.0 - dice_coef


#--------------------------------------------------------------------
# 模型保存
#--------------------------------------------------------------------
def save_model(model, name = None):
    prefix = './checkpoints/' + name + '.pth'
    # path = prefix + time.strftime('%m%d_%H:%M:%S.pth')
    # torch.save(model.state_dict(), path)
    torch.save(model.state_dict(), prefix)
    print('save model successs!!!')


#--------------------------------------------------------------------
# 模型加载
#--------------------------------------------------------------------
def load_model(model, name = None):
    prefix = './checkpoints/' + name + '.pth'
    # path = prefix + time.strftime('%m%d_%H:%M:%S.pth')
    # torch.save(model.state_dict(), path)
    try:
        model.load_state_dict(torch.load(prefix))
        print('load model successs!!!')
    except:
        print('load model fault!!!')
        pass

# --------------------------------------------------------------------
#  模型训练
#---------------------------------------------------------------------

def train(unet, train_dataset, val_dataset, use_gpu = False):


    if(use_gpu) == True:
        unet = unet.cuda()

    # 创建一个训练数据迭代器
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 定义损失函数
    criterion = torch.nn.BCELoss() # 二分类交叉熵

    # 定义优化器
    # optimizer = torch.optim.SGD(unet.parameters(),lr=0.01,momentum=0.9)
    # optimizer = torch.optim.Adam(unet.parameters())
    optimizer = torch.optim.Adam(unet.parameters(),lr=1e-5)
    # optimizer = torch.optim.Adadelta(unet.parameters(), lr=1.0, weight_decay=0.1) # loss下降比较快

    # 训练
    epochs = EPOCHS
    for epoch in range(epochs):

        print('开始第{}/{}轮训练'.format(epoch+1, epochs))
        epoch_loss = 0

        # 训练到第epoch轮
        for ii, (datas,labels) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):

            # 读入训练数据
            inputs, targets = Variable(datas), Variable(labels)
            targets = (targets > 0).float()

            # 使用GPU计算
            if(use_gpu):
                inputs = inputs.cuda()
                targets = targets.cuda()

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            preds = unet(inputs)

            # 计算loss
            bce_loss = criterion(preds.view(-1), targets.view(-1)) # loss1
            dice_loss = dice_cross(preds, targets) # loss2
            loss = alpha * bce_loss + (1.0 - alpha) * dice_loss  # 加权
            loss_data = loss.data
            print('loss={}'.format(loss_data)) # 打印loss

            # vis.plot('train_loss', loss_data) # 绘制loss曲线

            epoch_loss += loss_data # 统计loss

            # 梯度反向传播
            loss.backward()

            # 参数更新
            optimizer.step()

        # 一轮结束后计算epoch_loss
        epoch_loss = epoch_loss / len(train_dataloader)

        # 一轮结束后，在验证集上进行验证
        val_loss = val(unet, val_dataset, use_gpu)

        # 训练一轮后在visdom可视化一些参数
        print("本轮训练结束：Train_loss: {} Val_loss: {}".format(epoch_loss, val_loss))

        view.losses_hist[0].append(epoch_loss)  # 记录loss
        view.losses_hist[1].append(val_loss)  # 记录loss


        # 保存模型
        save_model(unet, save_model_name)


        # 显示loss曲线
        # view.show_loss(xlabel='Epochs', ylabel='Loss', title='train loss')

    # 保存训练loss
    view.save_loss(csv='./loss.csv')

# -----------------------------------------------------------------
#  模型验证
# -----------------------------------------------------------------

def val(unet, val_dataset, use_gpu=False):
    # 创建一个验证数据集迭代器
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 定义损失函数
    criterion = torch.nn.BCELoss()

    # 把模型设置为验证模式
    unet.eval()

    # 验证
    val_loss = 0
    for ii, (datas, labels) in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
        inputs, targets = Variable(datas), Variable(labels)
        targets = (targets > 0).float()
        if (use_gpu):
            inputs = inputs.cuda()
            targets = targets.cuda()
        preds = unet(inputs)


        # 计算测试loss
        dice_loss = dice_cross(preds, targets)
        bce_loss = criterion(preds.view(-1), targets.view(-1)) #计算loss
        loss = alpha * bce_loss + (1.0 - alpha) * dice_loss
        loss_data = loss.data
        print('loss={}'.format(loss_data)) # 打印loss

        # vis.plot('val_loss', loss_data)  # 绘制loss曲线

        val_loss += loss_data # 统计loss

    # 把模型恢复为训练模式
    unet.train()

    return val_loss / len(val_dataloader)


if __name__ == '__main__':

    # 实例化一个训练集对象
    train_dataset = Train_Data(TRAIN_DATASETS_ROOT, train=True)
    print(len(train_dataset))

    # 实例化一个验证集对象
    val_dataset = Train_Data(TRAIN_DATASETS_ROOT, train=False)
    print(len(val_dataset))

    # 实例化一个 Unet 模型
    unet = Unet2(3, 1).cuda()

    # 模型加载
    load_model(unet, save_model_name)

    # 实例化一个验证集对象
    val_dataset = Train_Data(TRAIN_DATASETS_ROOT, train=False)

    # 开始训练,使用gpu进行加速
    train(unet, train_dataset, val_dataset, use_gpu=True)

    # 保存模型
    # save_model(unet, save_model_name)
