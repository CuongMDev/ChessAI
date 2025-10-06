import torch
from torch import nn
import torch.nn.functional as F

from config.NetworkConfig import FILTER_CHANNEL, SE_CHANNELS

class SELayer(nn.Module):
    def __init__(self):
        super(SELayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(FILTER_CHANNEL, SE_CHANNELS)
        self.fc2 = nn.Linear(SE_CHANNELS, 2 * FILTER_CHANNEL)

    def forward(self, x):
        y = self.avg_pool(x)
        y = torch.flatten(y, 1)
        y = F.relu(self.fc1(y))
        y = self.fc2(y)

        W, B_bias = y.chunk(2, dim=1)
        Z = torch.sigmoid(W).unsqueeze(-1).unsqueeze(-1)
        B_bias = B_bias.unsqueeze(-1).unsqueeze(-1)

        return Z * x + B_bias
