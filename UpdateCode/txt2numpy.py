#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
数据集加载
Author   : Lum
Datetime : 2018/11/8
Filename : txt2numpy.py
'''

import os
import sys
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from sklearn.feature_extraction import image
import cv2

ROOT_PATH = '../segmentation_training_set/'
WHOLE_IMAGE_PATH = ROOT_PATH + 'images/'
COORDS_PATH = ROOT_PATH + 'masks/'

POLY_IMAGE_PATH = ROOT_PATH + 'images_poly/'
SAVE_PATH = ROOT_PATH + 'images_masks/'

if(not (os.path.exists(SAVE_PATH))):
    os.mkdir(SAVE_PATH)

image_filemames = os.listdir(WHOLE_IMAGE_PATH)
image_filemames = sorted(image_filemames, key=lambda x:int(x.split('.')[0][-1]) + int(x.split('.')[0][-2])*10)
image_ids = [filename.split('.')[0] for filename in image_filemames]


coords_filenames = os.listdir(COORDS_PATH)
coords_filenames = sorted(coords_filenames, key = lambda x: int(x.split('_')[0][-1]) + int(x.split('_')[0][-2])*10)


#---------------------------------------------------------
# 将 mask 坐标 txt文件 转换为 numpy 矩阵
#---------------------------------------------------------

def get_mask(filename, get_binary = False):

    # [0] 打开文件
    f = open(filename)

    # [1] 读取第一行的状态参数，创建mask矩阵
    line = f.readline()
    line = line.strip('\n')
    line = line.split(' ')
    width = int(line[0])
    height = int(line[1])
    mask = np.zeros((height, width), dtype=np.uint8)

    # [2] 读取每一行（标识每个像素归属的某个细胞核id，生成mask图）
    pid = 0
    line = f.readline()
    while line:
        line = line.strip('\n')
        mask[pid // width][pid % width] = int(line)
        line = f.readline()
        pid = pid + 1
    f.close()

    # [3] 如果需要获得二值mask
    if(get_binary):
        binary_mask = mask > 0
        binary_mask = binary_mask.astype(np.uint8)*255
        return binary_mask
    else:
        return mask.astype(np.uint8)


#---------------------------------------------------------
# 将 numpy 矩阵转换为 mask 坐标 txt文件
#---------------------------------------------------------
def save_mask(mask_array, txt_path):

    # # [0] 创建文件
    # if(not (os.path.exists(txt_path))):
    #     os.mknod(txt_path)

    # [1] 以写方式打开，第一行写下形状参数，'w'代表写
    mask_txt_file = open(txt_path, 'w')
    mask_txt_file.write('{} {}\n'.format(mask_array.shape[1], mask_array.shape[0]))

    # [2] 逐行写下mask矩阵的每一个元素值
    for v in mask_array.flat:
        mask_txt_file.write('{}\n'.format(v))

    mask_txt_file.close()



if __name__ == '__main__':

    # 实例化用例，数据预处理过程，转为易于操作的numpy格式
    for file in coords_filenames:
        filename = COORDS_PATH + file
        mask_img = get_mask(filename, get_binary = True)
        save = SAVE_PATH + file.split('.')[0] + '.png'
        print(save)
        cv2.imwrite(save, mask_img)

    # test_save = '../segmentation_training_set/test_save/'
    # test_mask_save = '../segmentation_training_set/test_mask/'
    #
    # test_save_file = os.listdir(test_save)
    # for file in test_save_file:
    #     filename = test_save + file
    #
    #     mask_img = get_mask(filename, get_binary = True) # 读取txt
    #     save = test_mask_save + file.split('.')[0] + '.png'
    #     print(save)
    #     cv2.imwrite(save, mask_img)
