import paddle
from x2paddle_code import FastPose
import os
import cv2 as cv
import numpy as np
import math
model = FastPose()  # 创建模型
model.eval()
model_dict = paddle.load("weight/SEResnte60")
model.set_state_dict(model_dict)
pkernel = np.ones((2,2),np.uint8)


def get_out_img(pathx):
    for af in os.listdir(pathx):
        print(af)
        imgy  =cv.imread(pathx+af)
        IMG_W = imgy.shape[1]
        IMG_H =imgy.shape[0]
        img = cv.resize(imgy, (1536, 1536), interpolation=cv.INTER_CUBIC)
        #img = cv.resize(imgy, (math.ceil(IMG_W / 32.0) * 32, math.ceil(IMG_H / 32.0) * 32), interpolation=cv.INTER_CUBIC) #如果显存足够大可以使用这个
        img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
        img_=np.array([img_],np.float32)/255.0
        inps = paddle.to_tensor(img_)
        out = model(inps).numpy()
        labimg = out[0, 1, :, :]

        labimg[labimg > 0.4] = 255
        slabimg = np.array(labimg, np.uint8)

        # cv.imshow('yimg', imgy)
        plabimg = cv.dilate(slabimg, pkernel, iterations=1)
        sp = cv.resize(plabimg, (IMG_W, IMG_H), interpolation=cv.INTER_CUBIC)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (4, 4))
        dilated = cv.dilate(sp, kernel)
        xxxout = cv.inpaint(imgy, dilated, 2.0, cv.INPAINT_TELEA)# 之前测试中效果更好但是提交时忘记使用了
        # cv.imshow('inpaint_img', xxxout)

        imgy[sp > 1] = [255, 255, 255]
        simg = cv.resize(img, (IMG_W, IMG_H), interpolation=cv.INTER_CUBIC)
        cv.imwrite('data/dehw_testB_dataset/result/'+af[:-3]+'png',imgy)
        # cv.imshow('data/dehw_testB_dataset/result/' + af[:-3] + 'png', imgy) # 提交代码方案
        # cv.waitKey(0)


if __name__ == '__main__':
    pathx= 'data/dehw_testB_dataset/image/'
    get_out_img(pathx)


