import time
import os
import paddle
import numpy as np
from tqdm import tqdm
import paddle.nn.functional as F
from x2paddle_code import FastPose
from data_train_paddleseg import data_read,DataLogger


def train():
    learning_rate = 1e-3
    train_dataset = data_read(train=True)
    train_loader =  paddle.io.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8, drop_last=True)
    model = FastPose()  #创建模型
    model.train()

    # model_dict = paddle.load("weight/SEResnte5")
    # model.set_state_dict(model_dict)
    opt = paddle.optimizer.Adam(learning_rate, parameters=model.parameters())
    MAX_EPOCH = 500
    for epoch in range(MAX_EPOCH):
        lossLogger = DataLogger()
        train_loader_desc = tqdm(train_loader)
        for i, (inps, labels) in enumerate(train_loader_desc):
            outputs = model(inps)  #前向传播，输出[P0, P1, P2]
            loss = F.mse_loss(outputs, labels)#model.get_loss(outputs, gt_boxes, gt_labels, gtscore=gt_scores,
                                  # anchors = ANCHORS,
                                  # anchor_masks = ANCHOR_MASKS,
                                  # ignore_thresh=IGNORE_THRESH,
                                  # use_label_smooth=False)        # 计算损失函数
            # lossLogger.update(loss.item(), inps.shape[0])
            loss.backward()    # 反向传播计算梯度
            opt.step()  # 更新参数
            opt.clear_grad()
            # train_loader_desc.set_description('loss: {loss:.8f}'.format(loss=lossLogger.avg,))

        if (epoch % 5 == 0) or (epoch == MAX_EPOCH -1):
            paddle.save(model.state_dict(), 'weight/SEResnte{}'.format(epoch))
        learning_rate=learning_rate*0.985

        print('loss:',str(lossLogger.avg))

if __name__ == '__main__':
   train()
