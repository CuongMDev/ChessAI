import torch
import torch.nn.functional as F
from torch import nn

from Agent.Network.ResidualBlock import ResidualBlock
from config.EnvConfig import INFO_SIZE, BOARD_SIZE, POLICY_OUT_CHANNEL, PIECES_ORDER
from config.NetworkConfig import FILTER_CHANNEL, VALUE_FC_SIZE, RES_LAYER_NUM, \
    FILTER_SIZE, EXTEND_INFO
from config.config import LABELS_MAP


class Network(nn.Module):
    def __init__(self, use_fp16=False, use_channels_last=False):
        super(Network, self).__init__()

        self.register_buffer('MASK_INDEX', torch.from_numpy(LABELS_MAP.mask_index), persistent=False)
        self.register_buffer('EXTEND_INFO', torch.from_numpy(EXTEND_INFO), persistent=False)

        self.use_channels_last = use_channels_last
        self.use_fp16 = use_fp16

        # common
        self.conv = nn.Conv2d(in_channels=INFO_SIZE + len(PIECES_ORDER) - 1 + len(EXTEND_INFO), out_channels=FILTER_CHANNEL, kernel_size=FILTER_SIZE, bias=False, padding='same') # -1 dấu .
        self.batch_norm = nn.BatchNorm2d(FILTER_CHANNEL)
        self.residual_blocks = nn.ModuleList([ResidualBlock() for _ in range(RES_LAYER_NUM)])

        # Policy
        self.pol_conv1 = nn.Conv2d(in_channels=FILTER_CHANNEL, out_channels=FILTER_CHANNEL, bias=False, kernel_size=FILTER_SIZE, padding='same')
        self.pol_batch_norm = nn.BatchNorm2d(FILTER_CHANNEL)
        self.pol_conv2 = nn.Conv2d(in_channels=FILTER_CHANNEL, out_channels=POLICY_OUT_CHANNEL, kernel_size=1)

        # Value
        self.val_conv = nn.Conv2d(in_channels=FILTER_CHANNEL, out_channels=32, bias=False, kernel_size=1)
        self.val_batch_norm = nn.BatchNorm2d(32)
        self.val_fc1 = nn.Linear(32 * BOARD_SIZE ** 2, VALUE_FC_SIZE)
        self.val_fc2 = nn.Linear(VALUE_FC_SIZE, 3)

    @torch.jit.ignore
    def one_hot(self, x):
        board = x[:, :, :BOARD_SIZE]
        half_move = x[:, :, -1] / 100

        board_one_hot = torch.stack([(board == i)
                             for i in range(1, len(PIECES_ORDER))]
                            ).transpose(0, 1)  # one hot chess piece
        info = torch.cat((x[:, :, BOARD_SIZE:-1], half_move.unsqueeze(-1)), dim=2).transpose(1, 2).unsqueeze(2).expand(x.size(0), INFO_SIZE, BOARD_SIZE, BOARD_SIZE)

        x = torch.cat([board_one_hot, info, self.EXTEND_INFO.unsqueeze(0).expand(x.shape[0], -1, -1, -1)], dim=1)
        if self.use_fp16:
            x = x.half()
        else:
            x = x.float()

        if self.use_channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        else:
            x = x.contiguous()

        return x

    def forward(self, x):
        x = self.one_hot(x)

        # common layers
        x = self.conv(x)
        x = self.batch_norm(x)
        x = F.relu(x)

        for layer in self.residual_blocks:
            x = layer(x)

        # policy layers
        x_pol = self.pol_conv1(x)
        x_pol = self.pol_batch_norm(x_pol)
        x_pol = F.relu(x_pol)
        x_pol = self.pol_conv2(x_pol)
        x_pol = torch.flatten(x_pol, start_dim=1)
        x_pol = torch.index_select(x_pol, dim=1, index=self.MASK_INDEX)

        # value layers
        x_val = self.val_conv(x)
        x_val = self.val_batch_norm(x_val)
        x_val = F.relu(x_val)
        x_val = torch.flatten(x_val, start_dim=1)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = self.val_fc2(x_val)

        return x_pol, x_val