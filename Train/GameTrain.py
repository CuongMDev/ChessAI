import gc
import multiprocessing
import math
import random
import time
from multiprocessing import Value

import chess.pgn
import numpy as np
import torch

from Agent.Agent import Agent
from Agent.ExperienceReplay import ExperienceReplay
from Env.GameState import GameState
from TrainProcess import play_with_agent, train_with_self
from config.config import EPISODE, GAME_EVALUATE, GAME_TRAIN_STEP, BATCH_SIZE, EPOCHS, WIN_UPDATE_PERCENT \
    , NUM_WORKERS, VALIDATION_SPLIT, PRETRAIN_FILE, PRETRAIN_GAME_ITERATION, PRETRAIN_EPOCHS, \
    PRETRAIN_MIN_VALUE_MOVE_NUMBER, OPENING_FILE, DEVICE, LABEL_SMOOTHING, LABELS_MAP


class GameTrain:
    def __init__(self, exp_min, device):
        self.agent = Agent(device)
        self.exp_min = exp_min
        self.experience_replay = ExperienceReplay()
        self.device = device

        self.pretrained = self.agent.load_checkpoint()

    def train_agent(self, batch_size: int, epochs: int, validation_split: float):
        develop_agent = self.agent.copy()

        train_loader, val_loader = self.experience_replay.get_all_data(batch_size, validation_split)
        train_loss, val_loss = develop_agent.fit(train_loader, epochs=epochs, val_loader=val_loader)

        self.experience_replay.delete_device_memory()

        return develop_agent, train_loss, val_loss

    def update_elo(self, agent, win_rate):
        e = 0.5
        agent.elo = agent.elo + 32 * (win_rate - e)

    def update_agent(self, develop_agent):
        all_openings = []
        with open(OPENING_FILE) as pgn_file:
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break  # Hết file
                all_openings.append(game)

        test_openings = [all_openings[0]] # start position
        test_openings.extend(random.sample(all_openings[1:], GAME_EVALUATE // 2 - 1))

        develop_agent.on_wait()
        develop_agent.memories.clear()
        self.agent.set_jit_mode('script')
        develop_agent.set_jit_mode('script')
        if DEVICE.type == 'cuda':
            torch.backends.cudnn.benchmark = False

        process = []
        results = multiprocessing.Array('i', 3)
        num_game = Value('i', GAME_EVALUATE)
        try:
            for worker in range(NUM_WORKERS):
                p = multiprocessing.Process(target=play_with_agent, args=(
                    self.agent.memories.get_share_memory(),
                    develop_agent.memories.get_share_memory(), results,
                    num_game, test_openings, worker))
                process.append(p)
                p.start()
            for p in process:
                p.join()
        except Exception:
            for p in process:
                p.terminate()
            raise Exception

        win_game = results[1]
        draw_game = results[2]
        loss_game = results[0]
        win_rate = (1 * win_game + 0.5 * draw_game) / GAME_EVALUATE

        self.update_elo(develop_agent, win_rate)
        print('elo: ', develop_agent.elo)

        print(f'win game: {win_game} - loss game: {loss_game} - draw_game: {draw_game}, win_rate: {win_rate}', flush=True)
        if win_rate > WIN_UPDATE_PERCENT:
            self.agent.on_stop()
            self.agent = develop_agent
            self.experience_replay.reset()

            self.agent.save_checkpoint()
            print('replaced network')
        else:
            develop_agent.on_stop()

    def pretrain(self):
        print('-------pretrain-------')
        if DEVICE.type == 'cuda':
            torch.backends.cudnn.benchmark = True

        start_time = time.time()
        all_games = []
        with open(PRETRAIN_FILE) as pgn_file:
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break  # Hết file
                all_games.append(game)

        for _ in range(PRETRAIN_EPOCHS):
            random.shuffle(all_games)
            for current_num_game, game in enumerate(all_games):
                game_state = GameState()
                cache = []  # state, the best policy, reward
                for chess_move in game.mainline_moves():
                    cache.append([game_state.get_train_board()])
                    pi = np.zeros(len(LABELS_MAP.labels_array))
                    legal_moves = game_state.get_legal_moves()
                    if len(legal_moves) > 1:
                        pi[legal_moves] = 1 + LABEL_SMOOTHING / (len(legal_moves) - 1)

                    move = game_state.real_uci_to_move(chess_move.uci())
                    game_state = game_state.perform_move(move)

                    pi[move] = 2 - LABEL_SMOOTHING

                    cache[-1].extend([pi, 1])

                reward = game_state.result
                for i in range(len(cache)):
                    if i < PRETRAIN_MIN_VALUE_MOVE_NUMBER:
                        cache[i][2] = 2 # 2 : not train value
                    elif (len(cache) - i) % 2:
                        cache[i][2] *= -reward
                    else:
                        cache[i][2] *= reward
                    self.experience_replay.add_experience(cache[i])

                if (current_num_game + 1) % PRETRAIN_GAME_ITERATION == 0 or current_num_game == len(all_games) - 1:
                    print(current_num_game // PRETRAIN_GAME_ITERATION, end=' ', flush=True)
                    develop_agent, _, _ = self.train_agent(BATCH_SIZE, 1, 0)
                    self.agent.on_stop()
                    self.agent = develop_agent
                    self.experience_replay.reset()
            print()

        del all_games
        gc.collect()

        end_time = time.time()
        print(f"{end_time - start_time:.2f}s")
        self.agent.save_checkpoint()

    def self_play_train(self):
        print('-------self_play-------')
        current_ep = 0
        self.agent.on_wait()
        start_time = time.time()

        for _ in range(math.ceil(EPISODE / GAME_TRAIN_STEP)):
            num_game = Value('i', min(EPISODE, current_ep + GAME_TRAIN_STEP) - current_ep)
            self.agent.memories.reset()
            self.agent.set_jit_mode('trace')
            if DEVICE.type == 'cuda':
                torch.backends.cudnn.benchmark = True

            process = []
            try:
                for worker in range(NUM_WORKERS):
                    p = multiprocessing.Process(target=train_with_self, args=(
                    self.agent.memories.get_share_memory(), self.experience_replay.get_share_memory(), num_game,
                    worker))
                    process.append(p)
                    p.start()
                for p in process:
                    p.join()
            except Exception:
                for p in process:
                    p.terminate()
                raise Exception

            current_ep += GAME_TRAIN_STEP
            print('\nEpisode: ', current_ep)
            if len(self.experience_replay) >= self.exp_min:
                develop_agent, train_loss, val_loss = self.train_agent(BATCH_SIZE, EPOCHS, VALIDATION_SPLIT)
                print('learning rate:', develop_agent.optimizer.param_groups[0]['lr'])
                print(f"train_loss: {train_loss}")
                print(f"val_loss: {val_loss}")

                self.update_agent(develop_agent)

            end_time = time.time()
            print(f"{end_time - start_time:.2f}s")
            start_time = time.time()

    def train(self):
        if not self.pretrained:
           self.pretrain()
        self.self_play_train()

