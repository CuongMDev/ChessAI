import torch

from Agent.Network.Network import Network
from Env.UciMapping import BOARD_SIZE
from config.NetworkConfig import INFO_SIZE
from config.config import SAVE_MODEL_PATH, MODEL_NAME

checkpoint = torch.load(SAVE_MODEL_PATH + MODEL_NAME, map_location='cpu')
model = Network()
model.load_state_dict(checkpoint['network_state_dict'])
model.eval()

dummy_input = torch.empty(1, BOARD_SIZE, BOARD_SIZE + INFO_SIZE, dtype=torch.int32)

torch.onnx.export(
    model,
    (dummy_input,),
    "saved_model/chess_model.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['board-state'],
    output_names=['policies-values'],
    dynamic_axes = {
        'board-state': {0: 'batch_size'},  # batch size là dynamic
        'policies-values': {0: 'batch_size'}  # output cũng theo batch
    }
)