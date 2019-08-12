from __future__ import absolute_import
import sys
import os

root_path = os.path.abspath(os.path.join(__file__, '..'))
sys.path.insert(0, os.path.join(root_path, 'lib'))


from utils.config import cfg
from dataset.dataset import Dataset
from model.network import NetWorks
from utils.loss import softmax_cross_entropy_loss
from utils.config import get_classes

import os
import torch.optim
import torch.cuda
import torch.utils.data
import torch.backends.cudnn
import time
import numpy as np


if __name__ == '__main__':

    cfg.ISTRAIN = True
    cfg_key = 'TRAIN' if cfg.ISTRAIN else 'TEST'

    classes = get_classes(True)
    num_classes = len(classes)

    if not cfg.TRAIN.CUDA and torch.cuda.is_available():
        print('You have cuda devices, so you should probably run with CUDA settings!')

    dataset = Dataset(classes)
    num_data =len(dataset)
    train_iters = int(num_data / cfg.TRAIN.BATCH_SIZE)
    print('total training iters in one epoch is {}'.format(train_iters))
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=cfg.ISTRAIN, num_workers=cfg[cfg_key].NUM_WORKS)

    net = NetWorks(num_classes=num_classes)
    print('networks initialize successfully')
    params = 0
    for param in net.parameters():
        params += torch.numel(param)
    print('The number of parameters in this networks is {:.2f}M.'.format(float(params) / 1e6))

    loss = softmax_cross_entropy_loss
    if cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(params=net.parameters(), lr=cfg.TRAIN.LEARNING_RATE, betas=(cfg.TRAIN.BETA1, 0.999), weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        raise NotImplementedError

    if cfg.TRAIN.CUDA:
        torch.backends.cudnn.benchmark = True
        net = net.cuda()

    output_dir = os.path.join(cfg.ROOT_DIR, 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if cfg.TRAIN.USE_TFBOARD:
        from tensorboardX import SummaryWriter
        logger = SummaryWriter("logs")

    dataiter = iter(dataloader)
    total_loss = 0
    start = time.time()
    for epoch in range(cfg.TRAIN.START_EPOCH, cfg.TRAIN.EPOCHS):
        for _iter in range(train_iters):
            try:
                data = next(dataiter)
            except:
                dataiter = iter(dataloader)
                data = next(dataiter)
            optimizer.zero_grad()
            input, labels = data[0], data[1]
            if cfg.TRAIN.CUDA:
                input = input.cuda()
                labels = labels.cuda()
            output = net(input)
            loss_value = loss(output, labels)
            loss_value.backward()
            optimizer.step()

            total_loss = loss_value.item()
            del loss_value

            if _iter % cfg.TRAIN.DISP_FREQ == 0:
                train_time = int(time.time() - start)
                start = time.time()
                print('[Iter {}/Epoch {}] --> Loss : {:.2f} for {:>2}mins{:<}secs'.format(_iter, epoch, total_loss, np.floor(train_time / 60).astype(np.int), (train_time % 60)))
                if cfg.TRAIN.USE_TFBOARD:
                    info = {'loss': total_loss}
                    logger.add_scalars("loss", info, (epoch + _iter / train_iters))

        torch.save(net.state_dict(), os.path.join(output_dir, 'net_{}.pkl'.format(epoch)))
    if cfg.TRAIN.USE_TFBOARD:
        logger.close()







