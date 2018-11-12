#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''

Author   : MG Studio
Datetime : 2018/11/8
Filename : Visdom.py
'''

import visdom
import time
import numpy as np

class Visualizer(object):
    def __init__(self, env='default', win='default', title='default', plot_together=True):

        self.vis = visdom.Visdom(env=env, port=8097)
        self.win = win
        self.title = title
        self.index = {}
        self.log_text = ''
        self.plot_together = plot_together # 在同一张图上绘图

    def plot(self, name, y):
        x = self.index.get(name, 0) # get(key, default) 返回指定键的值，若不存在则返回默认值
        self.index[name] = x
        if(not(self.plot_together)):
            self.win = name
            self.title = name

        self.vis.line(Y=np.array([y]),
                      X=np.array([x]),
                      win=self.win,
                      name=name,
                      opts=dict(title=self.title, showlegend=True),
                      update='append' if x > 0 else None)
        self.index[name] = x + 1


if __name__ == '__main__':

    # 将所有损失函数画在一张图中
    vis = Visualizer(plot_together=True)
    for epoch in range(10):
        loss1 = epoch + 1
        loss2 = epoch * epoch * 1/10
        vis.plot('loss1', loss1)
        vis.plot('loss2', loss2)
        time.sleep(0.1)
