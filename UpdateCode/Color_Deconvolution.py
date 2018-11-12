#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
颜色反卷积包括通过颜色分离特征
颜色反卷积算法是基于Lambert-Beer 定律,
颜色反卷积就是这样一个操作：
各个通道的灰度取负对数后乘以一个常矩阵的逆。
参考Tank's code 以及
参考网址：http://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_ihc_color_separation.html#sphx-glr-auto-examples-color-exposure-plot-ihc-color-separation-py
Author   : Lum
Datetime : 2018/11/8
Filename : Color_Deconvolution.py
'''

import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from skimage.color import rgb2hed
from matplotlib.colors import LinearSegmentedColormap
from io import BytesIO
import PIL
from Color_Deconvolution import *



root = '../segmentation_training_set/images/'
save_path = '../segmentation_training_set/images_hed/'


cmap_hema = LinearSegmentedColormap.from_list('mycmap', ['white', 'navy'])
cmap_dab = LinearSegmentedColormap.from_list('mycmap', ['white', 'saddlebrown'])
cmap_eosin = LinearSegmentedColormap.from_list('mycmap', ['darkviolet','white'])


def show_result(ihc_rgb, ihc_hed):
    fig, axes = plt.subplots(2, 2, figsize=(7,6), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(ihc_rgb)
    ax[0].set_title('Original image')

    # 苏木精
    ax[1].imshow(ihc_hed[:, :, 0], cmap=cmap_hema)
    ax[1].set_title("Hematoxylin")

    # 伊红
    ax[2].imshow(ihc_hed[:, :, 1], cmap=cmap_eosin)
    ax[2].set_title("Eosin")

    # 二氨基联苯胺（DAB）显示FHL2蛋白的IHC染色表达，其产生棕色
    ax[3].imshow(ihc_hed[:, :, 2], cmap=cmap_dab)
    ax[3].set_title("DAB")

    for a in ax.ravel():
        a.axis('off')

    # 布局显示
    fig.tight_layout()
    fig.show()

# -----------------------------------------------------------------
# rgb转换成hed,选取特定通道
# -----------------------------------------------------------------
def rgb_to_hed(channel = 0, rgb_file=None, hed_file=None, issave=False, isshow=False):
    ihc_rgb = io.imread(rgb_file)
    ihc_rgb = ihc_rgb[:, :, 0:3]

    # 将免疫组织化学（IHC）染色与苏木精复染色分开
    ihc_hed = rgb2hed(ihc_rgb)

    # 显示结果
    if isshow:
        show_result(ihc_rgb, ihc_hed)

    h, w = ihc_hed.shape[0], ihc_hed.shape[1]

    # 去除坐标轴边框
    plt.axis('off')
    fig = plt.gcf()
    fig.set_size_inches(w / 300, h / 300)  # dpi = 300, output = (w / dpi)*(h / dpi) pixels
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)

    plt.imshow(ihc_hed[:, :, channel], cmap=cmap_hema)

    if issave:
        fig.savefig(hed_file, format='png', transparent=True, dpi=300, pad_inches=0)

    # 申请缓冲地址
    buffer_ = BytesIO()  # using buffer,great way!
    # 保存在内存中，而不是在本地磁盘，注意这个默认认为你要保存的就是plt中的内容
    fig.savefig(buffer_, format='png', transparent=True, dpi=300, pad_inches=0)
    buffer_.seek(0)
    # 用PIL或CV2从内存中读取
    dataPIL = PIL.Image.open(buffer_)
    # 转换为nparrary，PIL转换就非常快了,data即为所需
    data = np.asarray(dataPIL)
    # 释放缓存
    buffer_.close()

    return data


if __name__ == '__main__':

    # 实例化一个反卷积操作
    for i in range(1, 16, 1):
        rgb_file = root + 'image{}{}.png'.format(i // 10, i % 10)
        hed_file = save_path + 'image{}{}.png'.format(i // 10, i % 10)

        data = rgb_to_hed(channel=0, rgb_file=rgb_file, hed_file=hed_file, issave=False, isshow=True)

        print(data.shape)
        plt.imshow(data)
        plt.show()
