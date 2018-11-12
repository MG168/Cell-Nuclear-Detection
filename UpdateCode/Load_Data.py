#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
数据集加载
Author   : MG Studio
Datetime : 2018/11/8
Filename : Load_Data.py
'''

import os
import sys
import numpy as np
from skimage import io
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.utils import data
from torchvision import transforms, utils
from torch.utils.data import DataLoader
import random
from txt2numpy import *
from Patch_set_create import *
from skimage.color import rgb2hed


# 获取所有图片的ID
whole_image_filenames = os.listdir(WHOLE_IMAGE_PATH)
whole_image_filenames = sorted(whole_image_filenames, key = lambda x: int(x.split('.')[0][-1]) + int(x.split('.')[0][-2])*10)
whole_image_ids = [filename.split('.')[0] for filename in whole_image_filenames]
# print(whole_image_ids)

patch_image_ids = []

for whole_img_id in whole_image_ids:
    patch_per_img_filenames = os.listdir(TRAIN_DATASETS_ROOT + whole_img_id + '/' + whole_img_id + '_patches/')
    patch_per_img_filenames = sorted(patch_per_img_filenames)
    patch_per_img_ids = [name.split('.')[-2] for name in patch_per_img_filenames]
    patch_image_ids.append(patch_per_img_ids)

# print(len(patch_image_ids))
# print(len(patch_image_ids[0]))


# ---------------------------------------------------------
# 用 Dataset 类封装训练集合验证集
# ---------------------------------------------------------
class Train_Data(data.Dataset):

    # 若train=True为训练集，否则为测试集
    def __init__(self, root, train = True):
        self.root = root

        # 划分训练集：验证集
        self.img_ids = []

        for i in range(len(patch_image_ids)):
            ids = patch_image_ids[i]

            if train :
                train_img_ids = ids[0:(int(0.8 * len(ids)))]

                self.img_ids.extend(train_img_ids)

            else:
                val_img_ids = ids[-int(0.2 * len(ids)):]
                self.img_ids.extend(val_img_ids)


    def __getitem__(self, index):

        # 加载img_patch
        img_id = self.img_ids[index]

        img_path = self.root + img_id.split('_')[-2] + '/{}_patches'.format(img_id.split('_')[-2]) + '/{}.png'.format(img_id)
        img_patch = io.imread(img_path)
        img_patch = img_patch[:,:,0:3]

        img_patch = transforms.ToTensor()(img_patch)

        # 加载mask_patch
        msk_path = self.root + img_id.split('_')[-2] + '/{}_masks'.format(img_id.split('_')[-2]) + '/{}_mask.txt'.format(img_id)
        msk_patch = get_mask(msk_path)
        msk_patch = torch.from_numpy(msk_patch.astype(np.int32)) # 转换为tensor
        msk_patch = msk_patch.unsqueeze(0) # 增加一个维度

        return img_patch, msk_patch

    def __len__(self):
        return len(self.img_ids)

# 显示批次大小
def show_batch(imgs):
    grid = utils.make_grid(imgs)
    plt.imshow(grid.numpy().transpose((1,2,0)))
    plt.title('Batch from dataloader')



if __name__ == '__main__':

    # 实例化一个训练集对象
    train_dataset = Train_Data(TRAIN_DATASETS_ROOT, train=True)
    print(len(train_dataset))

    # 实例化一个验证集对象
    val_dataset = Train_Data(TRAIN_DATASETS_ROOT, train=False)
    print(len(val_dataset))

    # 批量加载数据
    data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    for i, (batch_x, batch_y) in enumerate(data_loader):
        if (i < 2):
            print(i, batch_x.size(), batch_y.size())
            show_batch(batch_x) # 批量显示数据
            plt.axis('off')
            plt.show()
