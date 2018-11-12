#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
对输入的图片进行预测

Author   : MG Studio
Datetime : 2018/11/8
Filename : Predict_model.py
'''
import os
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision
from torchvision.transforms import ToPILImage
from torchvision import transforms
from skimage import measure, color
from Train_model import *
from Color_Deconvolution import *

TEST_SAVE_PATH = ROOT_PATH + 'predict_mask/'

if(not (os.path.exists(TEST_SAVE_PATH))):
    os.mkdir(TEST_SAVE_PATH)

# Size_pad = 16 # U_net
Size_pad = 32   # U_Net2

#------------------------------------------------------------------------------------------------
# 对图像边缘进行镜像扩展，使扩展后的图片长宽均为size的倍数
#-------------------------------------------------------------------------------------------------
def overlap_title(img, size=Size_pad):

    ih = img.shape[0]
    iw = img.shape[1]
    oh = (ih // size + int(ih % size > 0))*size
    ow = (iw // size + int(iw % size > 0))*size
    p_up = (oh - ih) // 2
    p_down = oh - ih - p_up
    p_left = (ow - iw) //2
    p_right = ow - iw - p_left
    pad_img = np.pad(img, ((p_up, p_down), (p_left, p_right), (0,0)), 'reflect')

    return pad_img, (p_up, p_up + ih, p_left, p_left + iw)


# 预测单张图片
def pred_one_img(path, model, use_cuda=False):
    # [1] 加载图像，取前3个通道
    # img_ = io.imread(path)
    img_ = rgb_to_hed(0, path)
    img_ = img_[:,:,0:3]
    img_, region = overlap_title(img_)

    # [2] 转化为channel-1st 的 4d tensor

    # img_ = img_.transpose(2, 0, 1)
    # img = torch.from_numpy(img_).float()
    # img = Variable(img.unsqueeze(0))

    img = transforms.ToTensor()(img_)
    img = Variable(img.unsqueeze(0))
    if use_cuda==True: img = img.cuda()
    print(img.shape)

    # [3] 输出预测概率图，并阈值化
    if use_cuda==True:
        binary_pred_msk = model(img).data.cpu()
    else:
        binary_pred_msk = model(img).data
    binary_pred_msk = binary_pred_msk > 0.5

    # [4] 转化为2-d array，并 crop 出和输入一致的区域
    binary_pred_msk = binary_pred_msk.squeeze().numpy()
    binary_pred_msk = binary_pred_msk[region[0]:region[1], region[2]:region[3]]

    return binary_pred_msk

######################################################
# 两个评估指标技术
######################################################

# Dice系数计算
def dice_coef(pred, target):
    smooth = 1 # 平滑项
    iflat = pred.flatten()
    tflat = target.flatten()
    intersection = (iflat * tflat).sum()
    dice_coef = (2. * intersection + smooth) / (iflat.sum()+ tflat.sum() + smooth)
    return dice_coef

# Ensemble Dice系数
def ensemble_dice_coef(nuclei_ids, true_msk):

    IntersectionArea = 0
    TotalMarkupArea = 0

    for i in range(1, nuclei_ids.max()):

        pred_bool_choose = nuclei_ids == i
        pred_nuclei_area = pred_bool_choose.sum()

        for j in range(1, true_msk.max()):

            true_bool_choose = true_msk == j
            true_nuclei_area = true_bool_choose.sum()

            AreaOfOverlap = (pred_bool_choose * true_bool_choose).sum()

            if (AreaOfOverlap > 0):
                IntersectionArea += AreaOfOverlap
                TotalMarkupArea += pred_nuclei_area + true_nuclei_area

    ensemble_dice_score = 2 * IntersectionArea / TotalMarkupArea

    return ensemble_dice_score


# 显示对比结果
def show_pred_result(image_id, orig_img, img_msk, binary_true_msk, binary_saved_msk, is_show=True):

    # 显示输出
    plt.figure(figsize=(7,7))

    plt.subplot(221)
    plt.title('orig_img')
    plt.imshow(orig_img)

    plt.subplot(222)
    plt.title('img_msk')
    plt.imshow(img_msk)

    plt.subplot(223)
    plt.title('binary_true_msk')
    plt.imshow(binary_true_msk)

    plt.subplot(224)
    plt.title('binary_saved_msk')
    plt.imshow(binary_saved_msk)

    # 保存结果
    plt.savefig(TEST_SAVE_PATH  + '{}_result.png'.format(image_id))
    # 显示结果
    if is_show:
        plt.show()


# 主函数
if __name__ == '__main__':

    use_cuda =  torch.cuda.is_available()
    # 加载训练好的模型
    if use_cuda:
        unet = Unet2(3, 1).cuda()
    else:
        unet = Unet2(3, 1)
    # 模型加载
    load_model(unet, save_model_name)

    # 对15张样本进行测试
    for i in range(1, 16, 1):
        image_id = 'image{}{}'.format(i//10, i%10) # 读入图片id

        # 预测输出的二值 msk
        binary_pred_msk = pred_one_img(WHOLE_IMAGE_PATH + '{}.png'.format(image_id), unet, use_cuda=use_cuda)
        print('binary_pred_msk:')
        print(binary_pred_msk.shape)

        # groundtruth 二值msk
        true_msk = get_mask(COORDS_PATH + '{}_mask.txt'.format(image_id))
        binary_true_msk = true_msk > 0
        binary_true_msk = binary_true_msk.astype(np.uint8)
        print('binary_true_msk:')
        print(binary_true_msk.shape)




        # 连通域分析，标记出每一个核:
        # 调用skimage.measure下的label函数实现二值图像的连通区域标记：
        nuclei_ids = measure.label(binary_pred_msk, connectivity=2)
        # print(nuclei_ids.shape)

        # 将标记矩阵转化为坐标 mask.txt
        save_mask(nuclei_ids, TEST_SAVE_PATH + '{}_pre_msk.txt'.format(image_id))

        # 加载mask.txt 检查是否保存正确
        binary_saved_msk = get_mask(TEST_SAVE_PATH + '{}_pre_msk.txt'.format(image_id))
        binary_saved_msk = binary_saved_msk > 0
        binary_saved_msk = binary_saved_msk.astype(np.uint8)
        print('binary_saved_msk:')
        print(binary_saved_msk.shape)



        # 将二值mask进行彩色量化
        color_msk = color.label2rgb(nuclei_ids, bg_label=0, bg_color=(1,1,1)) # bg_label,be_color 设置背景颜色
        print('color_msk:')
        print(color_msk.shape)
        io.imsave(TEST_SAVE_PATH + '{}_color_pred_msk.png'.format(image_id), color_msk)

        # 将原始输入和彩色mask合并
        orig_img = Image.open(WHOLE_IMAGE_PATH + '{}.png'.format(image_id))
        orig_img_a = orig_img.convert('RGBA') # 转成RGBA格式
        color_msk = Image.open(TEST_SAVE_PATH + '{}_color_pred_msk.png'.format(image_id))
        color_msk_a = color_msk.convert('RGBA') # 转成RGBA格式
        img_msk = Image.blend(orig_img_a, color_msk_a, 0.7) # 融合

        img_msk = np.array(img_msk, np.uint8) # 转成numpy数组
        io.imsave(TEST_SAVE_PATH + '{}_color_mask.png'.format(image_id), img_msk)


        # 显示最终效果
        show_pred_result('Image{}'.format(i), orig_img, img_msk, binary_true_msk, binary_saved_msk, is_show=True)

        # 计算Dice系数
        #print('Image', i, 'Dice:', dice_coef(binary_true_msk, binary_pred_msk))

        # 计算Ensemble Dice系数
        #print('Image', i, 'Ensemble Dice:', ensemble_dice_coef(nuclei_ids, true_msk))
