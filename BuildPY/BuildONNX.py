import sys

import torch

from Agent.Network.Network import Network
from config.config import SAVE_MODEL_PATH, MODEL_NAME, NETWORK_TYPE
from config.EnvConfig import FULL_INPUT_STATES

checkpoint = torch.load(SAVE_MODEL_PATH + MODEL_NAME, map_location='cpu')
model = Network(NETWORK_TYPE)
model.load_state_dict(checkpoint['network_state_dict'])
model.eval()

dummy_input = torch.empty(1, FULL_INPUT_STATES, dtype=torch.long)

try:
    torch.onnx.export(
        model,
        (dummy_input,),
        "saved_model/chess_model.onnx",
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['board-states'],
        output_names=['policies-values'],
        dynamic_axes={
            'board-states': {0: 'batch_size'},
            'policies-values': {0: 'batch_size'}
        }
    )
    print("success")
except Exception as e:
    print("Export failed:", e)
    input("Press Enter to exit...")  # giữ màn hình để đọc lỗi
    sys.exit(1)