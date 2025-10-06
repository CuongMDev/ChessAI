import torch
from torch import nn
from torch.nn import functional as F

from config.EnvConfig import POLICY_OUT_CHANNEL

class CnnPolicyHead(nn.Module):
    def __init__(self, in_channel, hidden_channel, filter_size):
        super().__init__()
        self.pol_conv1 = nn.Conv2d(in_channels=in_channel, out_channels=hidden_channel, bias=False, kernel_size=filter_size, padding='same')
        self.pol_batch_norm = nn.BatchNorm2d(hidden_channel)
        self.pol_conv2 = nn.Conv2d(in_channels=hidden_channel, out_channels=POLICY_OUT_CHANNEL, kernel_size=1)

    def forward(self, x):
        x = self.pol_conv1(x)
        x = self.pol_batch_norm(x)
        x = F.relu(x)
        x = self.pol_conv2(x)
        x = torch.flatten(x, start_dim=1)

        return x
