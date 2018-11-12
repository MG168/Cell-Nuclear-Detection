#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''

Author   : MG Studio
Datetime : 2018/11/8
Filename : Patch_set_create.py
'''

import os
from txt2numpy import *
from Elastic_Transformation import *
from Color_Deconvolution import *
from tqdm import tqdm



# 路径和参数设置
TRAIN_DATASETS_ROOT = '../patchwise_training_set_hed/'
PATCH_HEIGHT = 256
PATCH_WIDTH = 256
MAX_PATCHS_PER_IMG = 120


# 数据集扩增
def create_set():
    # 创建数据集目录根
    if(not (os.path.exists(TRAIN_DATASETS_ROOT))):
        os.mkdir(TRAIN_DATASETS_ROOT)

    # 为每一幅图片创建一个字目录，子目录下再创建两个目录
    for img_id in image_ids:
        if(not (os.path.exists(TRAIN_DATASETS_ROOT + img_id))):
            os.mkdir(TRAIN_DATASETS_ROOT + img_id)
        if(not (os.path.exists(TRAIN_DATASETS_ROOT + img_id + '/' + img_id + '_patches'))):
            os.mkdir(TRAIN_DATASETS_ROOT + img_id + '/' + img_id + '_patches')
        if (not (os.path.exists(TRAIN_DATASETS_ROOT + img_id + '/' + img_id + '_masks'))):
            os.mkdir(TRAIN_DATASETS_ROOT + img_id + '/' + img_id + '_masks')


    print('开始生成 patchs......')
    sys.stdout.flush()

    # 开始生成patchs和masks
    for i, img_id in tqdm(enumerate(image_ids), total=len(image_ids)):
        whole_mask = get_mask(COORDS_PATH + coords_filenames[i])
        whole_mask = np.expand_dims(whole_mask, 2)

        # rgb转换成hed,选取第一个通道
        whole_img = rgb_to_hed(0, WHOLE_IMAGE_PATH + image_filemames[i])
        whole_img = whole_img[:,:,0:3]

        # 将原图和mask合并成4通道
        whole_img_msk = np.concatenate((whole_img, whole_mask), axis=2)

        # 随机切割
        patchs1 = image.extract_patches_2d(whole_img_msk, (PATCH_WIDTH, PATCH_WIDTH), MAX_PATCHS_PER_IMG-20, i)

        # 进行弹性形变
        whole_img_msk = elastic_transfrom(whole_img_msk, alpha=2000, sigma=24)
        # 随机切割
        patchs2 = image.extract_patches_2d(whole_img_msk, (PATCH_WIDTH, PATCH_WIDTH), 20, i)

        # 合并
        patchs = np.vstack([patchs1, patchs2])

        for j, patch in enumerate(patchs):

            # 保存 img_patch 为 png 文件
            img_patch = patch[:,:,0:3]
            patch_save_path = TRAIN_DATASETS_ROOT + img_id + '/' + img_id + '_patches/'
            patch_save_name = img_id + '_patch{}{}.png'.format(j//10, j%10)
            io.imsave(patch_save_path + patch_save_name, img_patch)

            # 保存 msk_patch 为 txt 文件
            msk_patch = patch[:,:,3]
            msk_save_path = TRAIN_DATASETS_ROOT + img_id + '/' + img_id + '_masks/'
            msk_save_name = img_id + '_patch{}{}_mask.txt'.format(j // 10, j % 10)
            save_mask(msk_patch, msk_save_path + msk_save_name)





if __name__ == '__main__':

    # 实例化，创建数据集
    create_set()

    # 显示效果
    img_path_ = io.imread(TRAIN_DATASETS_ROOT + 'image03/image03_patches/image03_patch09.png')
    plt.imshow(img_path_)
    plt.show()
    msk_path_ = get_mask(TRAIN_DATASETS_ROOT + 'image01/image01_masks/image01_patch09_mask.txt')
    plt.imshow(msk_path_)
    plt.show()
