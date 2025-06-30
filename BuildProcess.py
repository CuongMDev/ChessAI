import PyInstaller.__main__
from config.config import SAVE_MODEL_PATH, MODEL_NAME

spec = [
    '--distpath', './GameProcess',
    '--add-data', f'{SAVE_MODEL_PATH + MODEL_NAME}:models',
    'game_socket.py'
]

PyInstaller.__main__.run(spec)