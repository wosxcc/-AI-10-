import torch
import torch.utils.data
from tqdm import tqdm
from SEresnet import createModel
import os
import cv2 as cv
import numpy as np
import math

net = createModel()
# 加载全部模型
net.load_state_dict(torch.load('../weight/Bmodel_272.pkl'))

import paddle


from x2paddle.convert import pytorch2paddle

inp = np.zeros((1,3,1536, 1536),np.float)
input_data = np.random.rand(1, 3, 224, 224).astype("float32")
# inp = np.ones((1,3, 1536, 1536), np.float32)
# inp = paddle.to_tensor(inp)

pytorch2paddle(net,
               save_dir="H:/",
               jit_type="trace",
               input_examples=[torch.tensor(input_data)])

net.eval()
# module (torch.nn.Module): PyTorch的Module。
# save_dir (str): 转换后模型的保存路径。
# jit_type (str): 转换方式。默认为"trace"。
# input_examples (list[torch.tensor]): torch.nn.Module的输入示例，list的长度必须与输入的长度一致。默认为None。

print("结束")
net.cuda()




pkernel = np.ones((2,2),np.uint8)


pathx= 'E:/dehw_testB_dataset/'
for af in os.listdir(pathx):
    imgy  =cv.imread(pathx+af)
    IMG_W = imgy.shape[1]
    IMG_H =imgy.shape[0]
    img = cv.resize(imgy, (1536, 1536), interpolation=cv.INTER_CUBIC)
    # img = cv.resize(imgy, (math.ceil(IMG_W / 32.0) * 32, math.ceil(IMG_H / 32.0) * 32), interpolation=cv.INTER_CUBIC)
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    inps = torch.from_numpy(img_).float().div(255.0).cuda()
    inps = inps.unsqueeze(0)
    out = net(inps).data.cpu().numpy()
    labimg = out[0, 1, :, :]

    labimg[labimg > 0.4] = 255
    slabimg = np.array(labimg, np.uint8)

    plabimg =cv.dilate(slabimg,pkernel,iterations=1)
    sp=cv.resize(plabimg, (IMG_W, IMG_H), interpolation=cv.INTER_CUBIC)
    #

    # cv.imshow('slabimg', slabimg)
    # cv.imshow('img', img)

    imgy[sp>1]=[255,255,255]
    # cv.imshow('slabimg', slabimg)
    # cv.imshow('plabimg', plabimg)
    # cv.imshow('sp', sp)
    #
    # print(sp.shape, img.shape)
    # cv.imshow('abimg', imgy)

    simg = cv.resize(img, (IMG_W, IMG_H), interpolation=cv.INTER_CUBIC)
    cv.imwrite('E:/result/'+af[:-3]+'png',imgy)
    # cv.waitKey()



#      slabimg = np.array(labimg, np.uint8)
#
#     plabimg = cv.dilate(slabimg, pkernel, iterations=1)
#     sp = cv.resize(plabimg, (IMG_W, IMG_H), interpolation=cv.INTER_CUBIC)
#     kernel = cv.getStructuringElement(cv.MORPH_RECT, (4, 4))
#     # eroded = cv.erode(RedThresh, kernel)  # 腐蚀图像
#     # dilated = cv.morphologyEx(sp, cv.MORPH_OPEN, kernel, iterations=1)
#     dilated = cv.dilate(sp, kernel)
#     #
#
#     # cv.imshow('slabimg', slabimg)
#     # cv.imshow('img', img)
#     xxxout = cv.inpaint(imgy, dilated, 2.0, cv.INPAINT_TELEA)
#
#     # cv.imshow('slabimg', slabimg)
#     # cv.imshow('plabimg', plabimg)
#     # cv.imshow('sp', sp)
#     #
#     # print(sp.shape, img.shape)
#     # cv.imshow('abimg', imgy)
#
#     # simg = cv.resize(img, (IMG_W, IMG_H), interpolation=cv.INTER_CUBIC)
#     print(af, 'E/dehw_testB_dataset/' + af[:-3] + 'png')
#     cv.imwrite('E/result/' + af[:-3] + 'png', xxxout)


