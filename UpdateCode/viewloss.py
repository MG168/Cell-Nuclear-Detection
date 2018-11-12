#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
绘制训练loss曲线和保存loss数据

Author   : MG Studio
Datetime : 2018/11/8
Filename : viewloss.py
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# loss可视化
class ViewLoss():

    def __init__(self, labels, losses_hist):
        self.labels = labels # 标签
        self.losses_hist = losses_hist # loss数据列表

    # 绘制loss曲线
    def show_loss(self, xlabel='Steps', ylabel='Loss', title='train loss'):
        for i, l_his in enumerate(self.losses_hist):
            plt.plot(l_his, label=self.labels[i]) # 绘制曲线

        # 绘制参数设置
        plt.legend(loc='best')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        # plt.ylim((0, 0.2))
        plt.savefig('{}.png'.format(title))
        plt.show()

    # 保存loss参数
    def save_loss(self, csv='./loss.csv'):
        losses_hist = np.asarray(self.losses_hist).T # 转成numpy在转置
        loss_data = pd.DataFrame(columns=self.labels, data=losses_hist) #加载到panda里面
        loss_data.to_csv(csv) #保存成csv格式

    # 将csv绘制成曲线图
    def plot_loss(self, csv='./loss.csv', xlabel='Steps', ylabel='Loss', title='train loss'):
        data = pd.read_csv(csv) # 读取csv文件
        data = np.array(data).T[1:]
        # print(data)

        for i, l_his in enumerate(data):
            plt.plot(l_his, label=self.labels[i]) # 绘制曲线

        # 绘制参数设置
        plt.legend(loc='best')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        # plt.ylim((0, 0.2))
        plt.savefig('{}.png'.format(title))
        plt.show()


if __name__ == '__main__':
    # 记录loss
    losses_hist = [[], []]
    # 绘制训练曲线
    labels = ['train_loss', 'val_loss']

    view = ViewLoss(labels, losses_hist)
    view.plot_loss(csv='loss_all.csv')
