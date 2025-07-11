from torch import nn
import torch.nn.functional as F

from Agent.Network.SELayer import SELayer
from config.NetworkConfig import FILTER_CHANNEL, FILTER_SIZE


class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(FILTER_CHANNEL, FILTER_CHANNEL, kernel_size=FILTER_SIZE, bias=False, padding='same')
        self.batch_norm1 = nn.BatchNorm2d(FILTER_CHANNEL)
        self.conv2 = nn.Conv2d(FILTER_CHANNEL, FILTER_CHANNEL, kernel_size=FILTER_SIZE, bias=False, padding='same')
        self.batch_norm2 = nn.BatchNorm2d(FILTER_CHANNEL)

        self.se = SELayer()

    def forward(self, x):
        in_x = x
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.se(x)

        x += in_x
        x = F.relu(x)

        return x
