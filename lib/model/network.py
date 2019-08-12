import torch
import torch.nn as nn
import torch.nn.functional as F


class NetWorks(nn.Module):
    def __init__(self, num_classes):
        super(NetWorks, self).__init__()

        self._num_classes = num_classes
        self.down_sample = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=True)
        self.dense_block_1 = DenseBlock(64)
        self.dense_block_2 = DenseBlock(64)
        self.pooling_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dense_block_3 = DenseBlock(64)
        self.dense_block_4 = DenseBlock(64)
        self.pooling_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dense_block_5 = DenseBlock(64)
        self.dense_block_6 = DenseBlock(64)
        self.pooling_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dense_block_7 = DenseBlock(64)
        self.dense_block_8 = DenseBlock(64)
        self.pooling_4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.linear_1 = nn.Linear(in_features=4096, out_features=1024)
        self.linear_2 = nn.Linear(in_features=1024, out_features=num_classes)

    def forward(self, input):
        out = self.down_sample(input)
        out = self.dense_block_1(out)
        out = self.dense_block_2(out)
        out = self.pooling_1(out)

        out = self.dense_block_3(out)
        out = self.dense_block_4(out)
        out = self.pooling_2(out)

        out = self.dense_block_5(out)
        out = self.dense_block_6(out)
        out = self.pooling_3(out)

        out = self.dense_block_7(out)
        out = self.dense_block_8(out)
        out = self.pooling_4(out)

        out = out.reshape(-1, 4096)

        out = self.linear_1(out)
        out = self.linear_2(out)
        out = torch.sigmoid(out)

        return out


class DenseBlock(nn.Module):
    def __init__(self, in_channels):
        super(DenseBlock, self).__init__()

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

        return dense_4

