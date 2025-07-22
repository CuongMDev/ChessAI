import torch

from Env.ExtendInfo import create_extend_info

# Model
EXTEND_INFO = create_extend_info()
FILTER_CHANNEL = 256
FILTER_SIZE = 3
RES_LAYER_NUM = 10
SE_CHANNELS = 32
VALUE_FC_SIZE = 128
EPOCHS = 1
VALIDATION_SPLIT = 0.02
VALIDATION_STEP = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_FP16 = DEVICE.type == 'cuda'