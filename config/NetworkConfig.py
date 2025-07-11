import torch

from Env.ExtendInfo import create_extend_info

# Model
INFO_SIZE = 6
EXTEND_INFO = create_extend_info()
FILTER_CHANNEL = 256
FILTER_SIZE = 3
RES_LAYER_NUM = 7
SE_CHANNELS = 32
VALUE_FC_SIZE = 256
EPOCHS = 1
VALIDATION_SPLIT = 0.02
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_DTYPE = torch.float
