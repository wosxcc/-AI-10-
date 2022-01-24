# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------

import os
from functools import reduce
import numpy as np
import cv2 as cv
import torch
import torch.utils.data as data
import random


class data_read(data.Dataset):
    def __init__(self, train=True, sigma=1,
                 scale_factor=(0.2, 0.3), rot_factor=40, label_type='Gaussian'):
        self.is_train = train           # training set or test set
        self.inputResH =512
        self.inputResW = 512
        self.outputResH = 512
        self.outputResW = 512
        self.scale_factor = scale_factor
        open_flie = open('train.txt','r')
        X_data_flie=open_flie.read().split('\n')
        open_flie.close()

        batch_img = []
        batch_lab = []
        couns_num =len(X_data_flie)-2 #  150000
        count = 0
        for a_data in X_data_flie:
            if count>couns_num:
                break
            batch_img.append(a_data.split(' ')[1])
            batch_lab.append(a_data.split(' ')[0])
            count += 1

        batch_img = np.array(batch_img)
        batch_lab = np.array(batch_lab)
        # train
        self.imgname_coco_train = batch_img[:-10]
        self.part_coco_train = batch_lab[:-10]

        self.size_train = self.imgname_coco_train.shape[0]-1

    def __getitem__(self, index):  # 如果在类中定义了__getitem__()方法，那么他的实例对象（假设为P）就可以这样P[key]取值。当实例对象做P[key]运算时，就会调用类中的__getitem__()方法。
        sf = self.scale_factor
        if self.is_train:

            imgname = self.imgname_coco_train[index]
            part = self.part_coco_train[index]

        inp_img = cv.imread(imgname)

        out_img = cv.imread(part, 0)
        # out_img = cv.resize(out_img, (self.outputResW, self.outputResH), interpolation=cv.INTER_CUBIC)
        # inp_img = cv.resize(inp_img, (self.inputResW, self.inputResH), interpolation=cv.INTER_CUBIC)

        left = random.randint(0, inp_img.shape[1]-1-self.outputResH)
        top = random.randint(0,  inp_img.shape[0]-1-self.outputResH)

        cinp = inp_img[top:top+self.outputResH, left:left+self.outputResH]
        cout = out_img[top:top+self.outputResH, left:left+self.outputResH]
        inp_img = cinp
        out_img = cout

        # #
        # cv.imshow('inp_imgy', inp_img)
        # cv.imshow('out_img', out_img * 255)
        # cv.waitKey()
        if (random.randint(0, 1)):  # 水平翻转
            # fp = random.randint(0, 1)
            inp_img = cv.flip(inp_img, 1)
            # out_img = cv.flip(out_img, fp)

        if (random.randint(0, 10) > 6):  # 随机遮盖
            randx = random.randint(10, 200)
            randy = random.randint(10, 200)
            ix = random.randint(0, self.inputResH - randx - 1)
            iy = random.randint(0, self.inputResH - randy - 1)
            # if (random.randint(0, 1)):
            #     inp_img[ix:ix+randx,iy:iy+randy]=[0,0,0]
            # else:
            inp_img[ix:ix + randx, iy:iy + randy] = [255, 255, 255]
            out_img[ix:ix + randx, iy:iy + randy] = 0

        if (random.randint(0, 10) > 6):  # 随机旋转
            angle = random.randint(-10, 10)
            ix = random.randint(200, 300)
            iy = random.randint(200, 300)
            bei = random.randint(8, 10) / 10
            M = cv.getRotationMatrix2D((ix, iy), angle, bei)

            inp_img = cv.warpAffine(inp_img, M, (self.inputResH, self.inputResW),borderValue=(255,255,255))  # ,flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE
            out_img = cv.warpAffine(out_img, M, (
            self.inputResH, self.inputResW))  # ,flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE

        if (random.randint(0, 10) > 6):  # 随机填充

            left = random.randint(1, 80)
            top = random.randint(1, 80)
            right = random.randint(1, 80)
            bottom = random.randint(1, 80)
            new_img_inp = np.ones((top + bottom + self.inputResH, left + right + self.inputResW, 3), np.uint8) * 255
            new_img_out = np.zeros((top + bottom + self.inputResH, left + right + self.inputResW), np.uint8)
            new_img_inp[top:top + self.inputResH, left:left + self.inputResW, :] = inp_img.copy()
            new_img_out[top:top + self.inputResH, left: left + self.inputResW] = out_img
            inp_img = cv.resize(new_img_inp, (self.inputResW, self.inputResH), interpolation=cv.INTER_CUBIC)
            out_img = cv.resize(new_img_out, (self.outputResW, self.outputResH), interpolation=cv.INTER_CUBIC)
        # cv.imshow('inp_imgy', inp_img)
        # cv.imshow('out_img', out_img * 255)
        # cv.waitKey()
        #
        # if (random.randint(0, 10) > 6):  # 随机裁剪
        #     left = random.randint(1, 80)
        #     top = random.randint(1, 80)
        #     right = random.randint(1, 80)
        #     bottom = random.randint(1, 80)
        #
        #     cinp = inp_img[top:-bottom, left:-right]
        #     cout = out_img[top:-bottom, left:-right]
        #     # inp_img[:left,:]=[255,255,255]
        #     # inp_img[:left, 0:top] = [255, 255, 255]
        #     # inp_img[right:,bottom:] =[255,255,255]
        #     # out_img[:left, 0:top] = 0
        #     # out_img[right:, bottom:] = 0
        #     inp_img = cv.resize(cinp, (self.inputResW, self.inputResH), interpolation=cv.INTER_CUBIC)
        #     out_img = cv.resize(cout, (self.outputResW, self.outputResH), interpolation=cv.INTER_CUBIC)

        img_ = inp_img[:, :, ::-1].transpose((2, 0, 1)).copy()
        inp = torch.from_numpy(img_).float().div(255.0)

        out = torch.from_numpy(out_img).float().unsqueeze(0)
        # metaData = generateSampleBox(imgname, bndbox, part, self.nJoints,
        #                              'hskj', sf, self, train=self.is_train,nJoints_coco=14)
        # inp, out, setMask = metaData

        setMask = torch.zeros(1, self.outputResH, self.outputResW)
        setMask[0].add_(1)

        return inp, out, setMask

    def __len__(self):
        if self.is_train:
            return self.size_train
        else:
            return self.size_val
# from tqdm import tqdm
# if __name__ == '__main__':
#
#
#     train_dataset = data_read(train=True)
#     train_loader_desc = tqdm(train_dataset)
#     for i, (inps, labels, setMask) in enumerate(train_loader_desc):
#         print()