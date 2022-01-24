class DataLogger(object):
    def __init__(self):
        self.clear()

    def clear(self):
        self.value = 0
        self.sum = 0
        self.cnt = 0
        self.avg = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.cnt += n
        self._cal_avg()

    def _cal_avg(self):
        self.avg = self.sum / self.cnt

# CUDA_VISIBLE_DEVICES=0

import torch
import torch.utils.data
import data_train2
from tqdm import tqdm
from SEresnet import createModel

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

nThreads = 2
trainBatch = 8
optMethod = 'adam'
Max_epoch = 400
LR = 1e-5
epoch = 0

loss = 0
def train(train_loader, m, criterion, optimizer):
    lossLogger = DataLogger()
    m.train()
    train_loader_desc = tqdm(train_loader)
    for i, (inps, labels, setMask) in enumerate(train_loader_desc):

        # print('得到数据',inps.shape, labels.shape, setMask.shape)

        inps = inps.cuda().requires_grad_()
        labels = labels.cuda()
        setMask = setMask.cuda()
        out = m(inps)
        # print('看看输出',out.shape)
        print(torch.sum(out.mul(setMask)-out),out.mul(setMask).shape,out.shape)
        loss = criterion(out.mul(setMask), labels)
        lossLogger.update(loss.item(), inps.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # TQDM
        train_loader_desc.set_description(
            'loss: {loss:.8f}'.format(
                loss=lossLogger.avg,
            )
        )

    train_loader_desc.close()

    return lossLogger.avg


def main():

    # Model Initialize
    m = createModel().cuda()
    # 加载全部模型
    # m.load_state_dict(torch.load('h:/weight1223/NBmodel_345.pkl'))



    criterion = torch.nn.MSELoss().cuda()

    if optMethod == 'rmsprop':
        optimizer = torch.optim.RMSprop(m.parameters(),
                                        lr=LR,
                                        momentum=0,
                                        weight_decay=0)
    elif optMethod == 'adam':
        optimizer = torch.optim.Adam(
            m.parameters(),
            lr=LR
        )
    else:
        raise Exception

    train_dataset = data_train2.data_read(train=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=trainBatch, shuffle=True, num_workers=nThreads, pin_memory=True)


    # Model Transfer
    m = torch.nn.DataParallel(m).cuda()

    # Start Training
    for i in range(Max_epoch):
        epoch = i

        print('############# Starting Epoch {} #############'.format(epoch))

        loss = train(train_loader, m, criterion, optimizer)

        print('Train-{idx:d} epoch | loss:{loss:.8f} '.format(
            idx=epoch,
            loss=loss,
        ))

        loss = loss
        m_dev = m.module

        torch.save(
            m_dev.state_dict(), 'h:/weight1223/NBmodel_{}.pkl'.format(epoch+1000))

        print('Valid-{idx:d} epoch | loss:{loss:.8f} '.format(
            idx=i,
            loss=loss,
        ))

if __name__ == '__main__':
    main()
