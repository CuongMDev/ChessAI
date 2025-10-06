import torch
from torch import nn

from Agent.Network.Attention.AttentionPolicyHead import AttentionPolicyHead
from Agent.Network.Attention.DeepNormInit import DeepNormInit
from Agent.Network.Attention.InputEmbedding import InputEmbedding
from Agent.Network.Attention.ResidualAttnBlock import ResidualAttnBlock
from Agent.Network.CNN.CnnPolicyHead import CnnPolicyHead
from Agent.Network.CNN.ResidualCnnBlock import ResidualCnnBlock
from config.EnvConfig import BOARD_SIZE, INPUT_PIECE_STATES, FULL_INPUT_STATES
from config.NetworkConfig import FILTER_CHANNEL, VALUE_FC_SIZE, \
    EXTEND_INFO, POW2_MASK, RESIDUAL_LAYER_NUM, DFF_DIM, ENCODING_DIM, NUM_HEADS, DROPOUT, \
    SMOLGEN_HIDDEN_CHANNELS, SMOLGEN_HIDDEN_SZ, SMOLGEN_GEN_SZ, FILTER_SIZE
from config.config import LABELS_MAP

class Network(nn.Module):
    def __init__(self, network_type='attention', use_fp16=False, use_channels_last=False):
        super(Network, self).__init__()

        self.register_buffer('MASK_INDEX', torch.from_numpy(LABELS_MAP.mask_index), persistent=False)
        self.register_buffer('EXTEND_INFO', torch.from_numpy(EXTEND_INFO), persistent=False)
        self.register_buffer("POW2_MASK", POW2_MASK, persistent=False)

        self.use_channels_last = use_channels_last
        self.use_fp16 = use_fp16

        # common
        if network_type == 'attention':
            init_fn = DeepNormInit(encoder_layers=RESIDUAL_LAYER_NUM)
            self.shared_logit_proj = nn.Linear(SMOLGEN_GEN_SZ, BOARD_SIZE ** 4)
            self.attn_blocks = [ResidualAttnBlock(FILTER_CHANNEL,
                    NUM_HEADS,
                    DFF_DIM,
                    DROPOUT,
                    (SMOLGEN_HIDDEN_CHANNELS, SMOLGEN_HIDDEN_SZ, SMOLGEN_GEN_SZ, self.shared_logit_proj),
                    init_fn,
                    BOARD_SIZE)
                  for _ in range(RESIDUAL_LAYER_NUM)]

            self.common = nn.Sequential(
                nn.Linear(FULL_INPUT_STATES + len(EXTEND_INFO), ENCODING_DIM), # position_encoding
                InputEmbedding(ENCODING_DIM, FILTER_CHANNEL, ffn_dim=DFF_DIM), # inp_emb
                *self.attn_blocks # attn blocks
            )
        elif network_type == 'cnn':
            self.common = nn.Sequential(
                nn.Conv2d(in_channels=FULL_INPUT_STATES + len(EXTEND_INFO), out_channels=FILTER_CHANNEL,
                                      kernel_size=FILTER_SIZE, bias=False, padding='same'),  # inp_emb
                nn.BatchNorm2d(FILTER_CHANNEL),
                *[ResidualCnnBlock() for _ in range(RESIDUAL_LAYER_NUM)] # cnn blocks
            )

        # Policy
        if network_type == 'attention':
            self.pol_head = AttentionPolicyHead(FILTER_CHANNEL, FILTER_CHANNEL, BOARD_SIZE)
        elif network_type == 'cnn':
            self.pol_head = CnnPolicyHead(FILTER_CHANNEL, FILTER_CHANNEL, FILTER_SIZE)

        # Value
        if network_type == 'attention':
            self.val_head = nn.Sequential(
                nn.Linear(FILTER_CHANNEL, 32),
                nn.ReLU(inplace=True),
                nn.Flatten(),
                nn.Linear(32 * BOARD_SIZE ** 2, VALUE_FC_SIZE),
                nn.ReLU(inplace=True),
                nn.Linear(VALUE_FC_SIZE, 3)
            )
        elif network_type == 'cnn':
            self.val_head = nn.Sequential(
                nn.Conv2d(in_channels=FILTER_CHANNEL, out_channels=32, bias=False, kernel_size=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Flatten(),
                nn.Linear(32 * BOARD_SIZE ** 2, VALUE_FC_SIZE),
                nn.ReLU(inplace=True),
                nn.Linear(VALUE_FC_SIZE, 3)
            )

    @torch.jit.ignore
    def extends(self, x):
        # chuẩn bị mask cho 64 bit
        board = x[:, :INPUT_PIECE_STATES]
        bits = (board.unsqueeze(-1) & self.POW2_MASK).to(torch.bool)  # (..., 64)
        board_extended = bits.view(*board.shape, BOARD_SIZE, BOARD_SIZE)  # reshape 8x8

        info = x[:, INPUT_PIECE_STATES:, None, None].expand(-1, -1, BOARD_SIZE, BOARD_SIZE).to(torch.float32)
        info[:, -1, :, :] /= 100 # half move / 100

        x = torch.cat([
            board_extended,
            info,
            self.EXTEND_INFO.unsqueeze(0).expand(x.shape[0], -1, -1, -1)
        ], dim=1)

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
        x = self.extends(x) # x: (B, C, H, W)
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)

        # common layers
        x = self.common(x)

        # policy layers
        x_pol = self.pol_head(x)
        x_pol = torch.index_select(x_pol, 1, self.MASK_INDEX)

        # value layers
        x_val = self.val_head(x)

        return x_pol, x_val