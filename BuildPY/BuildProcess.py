import PyInstaller.__main__
import os

from config.config import SAVE_MODEL_PATH, MODEL_ONNX_NAME

dll_folder = 'cuda_lib'
dlls = [
    f'{os.path.join(dll_folder, f)};.' for f in os.listdir(dll_folder) if f.endswith('.dll')
]

spec = [
    '--distpath', './GameProcess',

    # ✅ Thêm model ONNX
    '--add-data', f'{SAVE_MODEL_PATH + MODEL_ONNX_NAME}:saved_model',
    # ✅ Thêm Gaviota tablebase
    '--add-data', 'Gaviota;Gaviota'

    # ✅ Thêm tất cả DLL
] + sum([['--add-binary', dll] for dll in dlls], []) + [

    # ❌ Loại bỏ các module không cần
    '--exclude-module', 'torch',
    '--exclude-module', 'torchaudio',
    '--exclude-module', 'torchvision',
    '--exclude-module', 'triton',

    # ✅ File chính
    'Play/game_socket.py'
]

PyInstaller.__main__.run(spec)