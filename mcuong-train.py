import multiprocessing

from GameTrain import GameTrain
from config.config import EXP_MIN, DEVICE

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    game_train = GameTrain(EXP_MIN, DEVICE)
    game_train.train()