#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
弹性形变算法
高斯核大小计算公式： w = 2*int(truncate*sigma + 0.5) + 1
Author   : MG Studio
Datetime : 2018/11/8
Filename : Elastic_Transformation.py
'''

import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


# 弹性形变算法
def elastic_transfrom(image, alpha, sigma, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    # 高斯核
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="reflect", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="reflect", cval=0) * alpha
    dz = np.zeros_like(dx)

    # 计算矩阵坐标向量及索引，原通道排列是GRB
    x,y,z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1,1)), np.reshape(z+dz, (-1, 1))

    # 返回一个map的交叉索引
    return map_coordinates(image, indices, order=1, mode='constant').reshape(shape)



if __name__ == '__main__':

    # 数据预处理，选择一张图进行测试
    from txt2numpy import *

    img = io.imread(WHOLE_IMAGE_PATH + 'image03.png')
    img = img[:, :, 0:3]
    print('orig_img:')
    print(img.shape)
    plt.imshow(img)
    plt.show()

    msk = get_mask(COORDS_PATH + 'image03_mask.txt', get_binary=True)
    print('orig_msk:')
    print(msk.shape)
    plt.imshow(msk)
    plt.show()

    msk = np.expand_dims(msk, 2)
    img_msk = np.concatenate((img, msk), axis=2).astype(np.float64)


    # 实例化用例
    dst = elastic_transfrom(img_msk, alpha=2000, sigma=24)

    img_arg = dst[:,:,0:3].astype(np.uint8)
    print('img_arg:')
    print(img_arg.shape)
    plt.imshow(img_arg)
    plt.show()


    msk_arg = (dst[:,:,3]>0.5).astype(np.uint8)
    print('msk_arg:(作为第四通道与 image 同步变化获得)')
    print(msk_arg.shape)
    plt.imshow(msk_arg)
    plt.show()
