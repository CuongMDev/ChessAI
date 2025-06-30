import torch
from torch import nn
import torch.nn.functional as F

from Agent.Network.ResidualBlock import ResidualBlock
from Env.UciMapping import POLICY_OUT_CHANNEL
from config.config import BOARD_SIZE, PIECES_ORDER, FILTER_CHANNEL, VALUE_FC_SIZE, RES_LAYER_NUM, \
    INFO_SIZE, LABELS_MAP, FILTER_SIZE, EXTEND_INFO, MODEL_DTYPE


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.register_buffer('UCI_LABELS_MAP', torch.from_numpy(LABELS_MAP.create_uci_labels_mask()), persistent=False)
        self.register_buffer('EXTEND_INFO', torch.from_numpy(EXTEND_INFO), persistent=False)

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

        board_one_hot = torch.stack([(board == i)
                             for i in range(1, len(PIECES_ORDER))]
                            ).transpose(0, 1)  # one hot chess piece
        info = x[:, :, BOARD_SIZE:].transpose(1, 2).unsqueeze(2).expand(x.size(0), INFO_SIZE, BOARD_SIZE, BOARD_SIZE)

        x = torch.cat([board_one_hot, info, self.EXTEND_INFO.unsqueeze(0).expand(x.shape[0], -1, -1, -1)], dim=1).to(MODEL_DTYPE)
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
        x_pol = x_pol[:, self.UCI_LABELS_MAP]

        # value layers
        x_val = self.val_conv(x)
        x_val = self.val_batch_norm(x_val)
        x_val = F.relu(x_val)
        x_val = torch.flatten(x_val, start_dim=1)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = self.val_fc2(x_val)

        return x_pol, x_val