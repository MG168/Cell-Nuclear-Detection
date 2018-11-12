#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
细胞分割可视化操作
Author   : MG Studio
Datetime : 2018/8/3 19:37
Filename : main.py
'''

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QWidget, QApplication, QMessageBox, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from Ui_Main import Ui_MainWindow
import sys
import cv2
from skimage.io import imread
from U_net import Unet
from Train_model import *
from Predict_model import *


# 读入RGB图像并转为QImage
def rgb2QImage(imgpath):

    img = cv2.imread(imgpath, 1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 颜色空间转换，在opencv中，其默认的颜色制式排列是BGR而非RGB

    height, width, bytesPerComponent = img_rgb.shape
    # bytesPerLine = 3 * width

    qimage = QImage(img_rgb.data, width, height, QImage.Format_RGB888)  # 根据图像宽高来构造一幅图像，程序会自动根据图像格式对齐图像数据。

    return qimage



def pred_process(imgname, model):
    binary_pred_msk = pred_one_img(imgname, model, use_cuda=True)
    nuclei_ids = measure.label(binary_pred_msk, connectivity=2)  # measure子模块下的label（）函数来实现连通区域标记，1代表4邻接，2代表8邻接
    id_count = nuclei_ids.max()
    print(nuclei_ids.max())
    color_msk = color.label2rgb(nuclei_ids, bg_label=0, bg_color=(1, 1, 1))  # bg_label,be_color 设置背景颜色
    print(color_msk.shape)

    plt.imsave('color_pred_mask.png', color_msk)

    orig_img = Image.open(imgname)
    orig_img = orig_img.convert('RGBA')

    color_msk = Image.open('color_pred_mask.png')

    color_msk = color_msk.convert('RGBA')
    img_msk = Image.blend(orig_img, color_msk, 0.7)

    plt.imsave('img_msk.png', img_msk)

    return id_count


# 封装UI
class Action(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):

        super(Action, self).__init__(parent)
        # 初始化UI
        self.setupUi(self)

        self.statusBar().showMessage('请输入要预测的图片！')  # 状态栏显示

        # 连接加载文件槽函数
        self.pushButton_load.clicked.connect(self.pushButton_load_clicked)

        # 连接预测输出槽函数
        self.pushButton_pred.clicked.connect(self.pushButton_pred_clicked)


        # 加载训练好的模型
        self.Unet = Unet2(3, 1).cuda()

        # 模型加载
        load_model(self.Unet, save_model_name)


    # 关闭窗口弹出显示框
    def closeEvent(self, event):

        realy = QMessageBox.question(self, '提示', '确认退出吗', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if realy == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    # 打开文件槽函数
    @pyqtSlot()
    def pushButton_load_clicked(self):
        # 打开文件
        self.filename = QFileDialog.getOpenFileName(self, "OpenFile", ".", "Image Files(*.jpg *.jpeg *.png)")[0]
        print("Image Path", self.filename)

        if len(self.filename):

            self.imgname = self.filename
            self.image = rgb2QImage(self.imgname) # RGB图转QImage
            self.label_img.setPixmap(QPixmap.fromImage(self.image))
            self.label_img.resize(self.image.width(), self.image.height())

    # 预测输出槽函数
    @pyqtSlot()
    def pushButton_pred_clicked(self):

        self.statusBar().showMessage('正在预测中，请稍后...')  # 状态栏显示
        try:
            id_count = pred_process(self.imgname, self.Unet)  # 输入模型进行预测

            self.image2 = rgb2QImage('img_msk.png')

            self.label_pred.setPixmap(QPixmap.fromImage(self.image2))
            self.label_pred.resize(self.image2.width(), self.image2.height())
            self.statusBar().showMessage('预测成功！发现细胞核大约{}个。'.format(id_count))

        except:
            self.statusBar().showMessage('预测失败！！！')
            pass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    act = Action()   # 实例化action
    act.show()
    sys.exit(app.exec_())
