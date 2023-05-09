import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Perform 3D conv, BN and elu activation to go from in_channels to out_channels
    """

    def __init__(self, in_channels, out_channels, use_batch_norm=True, k_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        midchann = int((in_channels + out_channels) / 2)

        self.conv3d_a = nn.Conv3d(in_channels=in_channels, out_channels=midchann, kernel_size=k_size,
                                  stride=stride, padding=padding)
        self.conv3d_b = nn.Conv3d(in_channels=midchann, out_channels=out_channels, kernel_size=k_size,
                                  stride=stride, padding=padding)
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_a = nn.BatchNorm3d(num_features=midchann)
            self.batch_norm_b = nn.BatchNorm3d(num_features=out_channels)

    def forward(self, x):
        x = self.conv3d_a(x)
        if self.use_batch_norm:
            x = self.batch_norm_a(x)
        x = F.elu(x)
        x = self.conv3d_b(x)
        if self.use_batch_norm:
            x = self.batch_norm_b(x)
        x = F.elu(x)
        return x


class RMSDModel(nn.Module):
    """
    Perform 3D conv, BN and elu activation to go from in_channels to out_channels
    """

    def __init__(self, channels=(5, 128, 256, 512)):
        super(RMSDModel, self).__init__()
        self.convs = nn.ModuleList()
        for i, (prev, next) in enumerate(zip(channels, channels[1:])):
            self.convs.append(ConvBlock(prev, next, use_batch_norm=i < 2))
            self.convs.append(nn.MaxPool3d(kernel_size=3, stride=2))

        # self.fc = nn.ModuleList()
        # self.fc.append()
        self.fc1 = nn.Linear(channels[-1], 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # augment channels, reduce size
        for i, block in enumerate(self.convs):
            # print("block : ", block._get_name())
            # print("preop : ", x.shape)
            x = block(x)
            # print("postop : ", x.shape)
            # try:
            #     print("I used BN : ", block.use_batch_norm)
            # except:
            #     pass
        # print("done")

        if not all([1 == i for i in x.shape[-3:]]):
            x = nn.AvgPool3d(kernel_size=x.shape[-3])(x)

        # Then reshape to get just the embedding and go through FC
        x = torch.reshape(x, x.shape[:2])
        return self.fc2(F.relu(self.fc1(x)))
