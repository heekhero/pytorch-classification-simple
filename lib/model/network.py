import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.config import cfg


class NetWorks(nn.Module):
    def __init__(self, num_classes):
        super(NetWorks, self).__init__()

        self._num_classes = num_classes
        self.down_sample = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=True)
        self.pooling_0 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dense_block_1 = DenseBlock(6, 64)
        self.trans_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0, stride=1, bias=True)
        self.pooling_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.drop_1 = nn.Dropout(p=0.02)

        self.dense_block_2 = DenseBlock(12, 64)
        self.trans_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0, stride=1, bias=True)
        self.pooling_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.drop_2 = nn.Dropout(p=0.02)

        self.dense_block_3 = DenseBlock(16, 64)
        self.trans_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0, stride=1, bias=True)
        self.pooling_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.drop_3 = nn.Dropout(p=0.02)

        # self.ad_pooling = nn.AdaptiveAvgPool2d(int(cfg.TRAIN.RESIZE/32))
        self.drop_out1 = nn.Dropout(p=0.5)
        self.linear1 = nn.Linear(in_features=(int(cfg.TRAIN.RESIZE/32) ** 2) * 64, out_features=1024)

        self.drop_out2 = nn.Dropout(p=0.3)
        self.linear2 = nn.Linear(in_features=1024, out_features=num_classes)

    def forward(self, input):

        out = self.down_sample(input)
        out = self.pooling_0(out)

        out = self.dense_block_1(out)
        out = self.trans_1(out)
        out = self.pooling_1(out)
        # out = self.drop_1(out)

        out = self.dense_block_2(out)
        out = self.trans_2(out)
        out = self.pooling_2(out)
        # out = self.drop_2(out)

        out = self.dense_block_3(out)
        out = self.trans_3(out)
        out = self.pooling_3(out)
        # out = self.drop_3(out)

        # out = self.ad_pooling(out)
        out = out.reshape(-1, (int(cfg.TRAIN.RESIZE/32) ** 2) * 64)

        out = self.drop_out1(out)
        out = self.linear1(out)

        out = self.drop_out2(out)
        out = self.linear2(out)
        out = torch.sigmoid(out)

        return out


class DenseBlock_6(nn.Module):
    def __init__(self, in_channels):
        super(DenseBlock_6, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, stride=1, bias=True)
        self.bn1 = nn.BatchNorm2d(num_features=in_channels)

        self.conv2 = nn.Conv2d(in_channels= 2 * in_channels, out_channels= in_channels, kernel_size=3, padding=1, stride=1, bias=True)
        self.bn2 = nn.BatchNorm2d(num_features= in_channels)

        self.conv3 = nn.Conv2d(in_channels=3 * in_channels, out_channels= in_channels, kernel_size=3, padding=1, stride=1, bias=True)
        self.bn3 = nn.BatchNorm2d(num_features= in_channels)

        self.conv4 = nn.Conv2d(in_channels=4 * in_channels, out_channels=in_channels, kernel_size=3, padding=1, stride=1, bias=True)
        self.bn4 = nn.BatchNorm2d(num_features=in_channels)

        self.conv5 = nn.Conv2d(in_channels=5 * in_channels, out_channels=in_channels, kernel_size=3, padding=1, stride=1, bias=True)
        self.bn5 = nn.BatchNorm2d(num_features=in_channels)

        self.conv6 = nn.Conv2d(in_channels=6 * in_channels, out_channels=in_channels, kernel_size=3, padding=1, stride=1, bias=True)
        self.bn6 = nn.BatchNorm2d(num_features=in_channels)

    def forward(self, input_0):
        dense_0 = self.bn1(F.relu(self.conv1(input_0)))
        input_1 = torch.cat([input_0, dense_0], 1)

        dense_1 = self.bn2(F.relu(self.conv2(input_1)))
        input_2 = torch.cat([input_1, dense_1], 1)

        dense_2 = self.bn3(F.relu(self.conv3(input_2)))
        input_3 = torch.cat([input_2, dense_2], 1)

        dense_3 = self.bn4(F.relu(self.conv4(input_3)))
        input_4 = torch.cat([input_3, dense_3], 1)

        dense_4 = self.bn5(F.relu(self.conv5(input_4)))
        input_5 = torch.cat([input_4, dense_4], 1)

        out = self.bn6(F.relu(self.conv6(input_5)))

        return out


class DenseBlock(nn.Module):
    def __init__(self, num_blocks, base_feat):
        super(DenseBlock, self).__init__()
        self._num_blocks = num_blocks
        self._base_feat = base_feat
        for i in range(1, self._num_blocks + 1):
            setattr(self, 'conv_' + str(i), nn.Conv2d(in_channels=self._base_feat * i, out_channels=self._base_feat, kernel_size=3, padding=1, stride=1, bias=True))
            setattr(self, 'bn_' + str(i), nn.BatchNorm2d(num_features=self._base_feat, affine=True, track_running_stats=True))

    def forward(self, input):
        for i in range(1, self._num_blocks + 1):
            bn = eval('self.bn_' + str(i))
            conv = eval('self.conv_' + str(i))
            out = bn(F.relu(conv(input)))
            input = torch.cat([input, out], 1)
        return out

