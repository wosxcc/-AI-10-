import cv2 as cv
import os
import numpy as np


pathx = 'data/dehw_train_dataset/images/'
pathy = 'data/dehw_train_dataset/gts/'
# pathc = 'dehw_train_dataset/imgcha/'
pathci = 'data/dehw_train_dataset/imgclasss0/'

def read_file_img():


    save_txt = ''
    countx = 0
    for afile in os.listdir(pathx):
        imgx = cv.imread(pathx+afile)
        imgy = cv.imread(pathy + afile[:-3]+'png')

        imgc= imgy-np.array(imgx,np.int)
        # np.save(pathc+afile[:-3]+'npy',imgc)
        imgc[imgc<50]=0
        imgsum = np.sum(imgc,axis=2)
        # print(imgsum.shape)
        imgc[imgsum>50]=[255,255,255]

        cv.imwrite(pathci+afile,np.array(imgc,np.uint8))
        save_txt+=pathx+afile+' '+pathci+afile  #保存为训练中间文件

        # cv.imshow('imgc', np.array(imgc,np.uint8))
        # cv.imshow('imgx',imgx)
        # cv.waitKey()
        print(countx)
        countx+=1
    opentxt = open('train.txt', 'w')
    opentxt.write(save_txt)
    opentxt.close()

read_file_img()