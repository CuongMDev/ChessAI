import torch

from Env.ExtendInfo import create_extend_info
from config.EnvConfig import BOARD_SIZE

# Attn
NUM_HEADS = 16
ENCODING_DIM = 128
DFF_DIM = 1024
DROPOUT = 0.0

SMOLGEN_HIDDEN_CHANNELS = 32
SMOLGEN_HIDDEN_SZ = 256
SMOLGEN_GEN_SZ = 256

# CNN
FILTER_SIZE = 3
SE_CHANNELS = 32

# Model
EXTEND_INFO = create_extend_info()
RESIDUAL_LAYER_NUM = 15
FILTER_CHANNEL = 512
VALUE_FC_SIZE = 128
EPOCHS = 1
VALIDATION_SPLIT = 0.02
VALIDATION_STEP = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_FP16 = DEVICE.type == 'cuda'

POW2_MASK = 1 << torch.arange(BOARD_SIZE ** 2, dtype=torch.long)
