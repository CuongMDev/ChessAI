import copy
import os
from threading import Thread, Event

import torch
from torch import optim, nn

from Agent.CustomLearningRate import CustomLearningRateSchedule
from Agent.Network.Network import Network
from Agent.AgentMemories import AgentMemories
from config.NetworkConfig import MODEL_DTYPE, INFO_SIZE
from config.config import LEARNING_RATE, SAVE_MODEL_PATH, L2_CONST, DECAY_RATE, \
    NUM_WORKERS, MIN_EVALUATE_COUNT, MODEL_NAME, BOARD_SIZE, MOMENTUM, LOSE_WEIGHTS, \
    MAX_GRAD_NORM


class Agent:
    def __init__(self, device, num_worker=NUM_WORKERS, min_evaluate_count=MIN_EVALUATE_COUNT):
        self.network = Network().to(device)
        self.memories = AgentMemories(num_workers=num_worker, min_evaluate_count=min_evaluate_count)
        self.stop_event = Event()
        self.wait_process = Thread(target=self.wait_and_evaluate, args=(self.memories.need_evaluate, self.stop_event))
        self.wait_process.daemon = True

        self.example_inputs_size = None
        self.network_jit = None
        self.jit_mode = None

        self.elo = 0

        self.optimizer = optim.SGD(self.network.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=L2_CONST, nesterov=True)
        self.scheduler = CustomLearningRateSchedule(
            optimizer=self.optimizer,
            initial_lr=LEARNING_RATE,
            decay_rates=DECAY_RATE,
        )
        self.loss_policy = nn.CrossEntropyLoss()
        self.loss_value = nn.CrossEntropyLoss(ignore_index=3)
        self.device = device

    def copy(self):
        new_network = Agent(self.device)

        new_network.network.load_state_dict(self.network.state_dict())
        new_network.optimizer.load_state_dict(self.optimizer.state_dict())
        new_network.scheduler.load_state_dict(self.scheduler.state_dict())
        new_network.elo = self.elo

        return new_network

    def fit(self, train_loader, epochs, val_loader=None):
        train_loss = []
        val_loss = []
        for epoch in range(epochs):
            self.network.train()

            total_loss = 0
            total_samples = 0
            for inp in train_loader:
                self.optimizer.zero_grad()

                loss = self.get_loss(inp)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.network.parameters(), MAX_GRAD_NORM)

                total_loss += loss.item() * len(inp[0])
                total_samples += len(inp[0])

                self.optimizer.step()

            train_epoch_loss = total_loss / total_samples
            train_loss.append(train_epoch_loss)

            if val_loader is not None:
                val_epoch_loss = self.validate(val_loader)
                val_loss.append(val_epoch_loss)

        return train_loss, val_loss  # Trả về giá trị loss cuối cùng (float)

    def validate(self, val_loader):
        self.network.eval()

        total_loss = 0
        total_samples = 0

        with torch.no_grad():
            for inp in val_loader:
                loss = self.get_loss(inp)

                total_loss += loss.item() * len(inp[0])
                total_samples += len(inp[0])

        avg_loss = total_loss / total_samples
        return avg_loss  # Trả về giá trị loss cuối cùng (float)

    def get_loss(self, inp):
        state, policy_target, value_target = inp

        policy, value = self.network(state)

        policy_mask = policy_target != 0
        policy_target_no_mask = policy_target.clone()
        policy_target_no_mask[policy_mask] -= 1  # Bỏ thông tin mask
        policy[~policy_mask] = -1e9  # bỏ phần không hợp lệ

        loss_policy = self.loss_policy(policy, policy_target_no_mask)
        loss_value = self.loss_value(value, value_target + 1)
        loss = loss_policy * LOSE_WEIGHTS[0] + loss_value * LOSE_WEIGHTS[1]

        return loss

    def evaluate(self, state):
        with torch.no_grad():
            if self.jit_mode == 'trace' and len(state) != self.example_inputs_size[0]:
                self.set_jit_mode('script')

            output = self.network_jit(state)

        return output

    def set_jit_mode(self, mode):
        self.jit_mode = mode
        if self.jit_mode == 'trace':
            self.example_inputs_size = (min(self.memories.num_current_worker.value, MIN_EVALUATE_COUNT), BOARD_SIZE, BOARD_SIZE + INFO_SIZE)
            self.network_jit = torch.jit.trace(
                copy.deepcopy(self.network).to(MODEL_DTYPE),
                example_inputs=torch.empty(self.example_inputs_size, dtype=torch.int32, device=self.device),
            )
        else:
            self.network_jit = torch.jit.script(copy.deepcopy(self.network).to(MODEL_DTYPE))

    def wait_and_evaluate(self, event, stop_event):
        while True:
            event.wait()
            if stop_event.is_set():
                break

            self.evaluate_all()

    def evaluate_all(self):
        need_evaluate_indexes = [i for i, event in enumerate(self.memories.share_valid_data) if not event.is_set()][:MIN_EVALUATE_COUNT] # is needing evaluated

        policies, value = self.evaluate(self.memories.share_input[need_evaluate_indexes].to(self.device))
        value = value.float().cpu()

        for i, index in enumerate(need_evaluate_indexes):
            policies_i = policies[i, self.memories.share_input_mask[index, :self.memories.share_input_mask_size[index]]]
            self.memories.share_output_pol[index, :self.memories.share_input_mask_size[index]] = policies_i.float().cpu()
            self.memories.share_output_val[index] = value[i]
            self.memories.share_valid_data[index].set()

        self.memories.change_need_predict_count(-len(need_evaluate_indexes))
        self.memories.need_evaluate.clear()
        self.memories.change_need_predict_count(0) # Check if need predict

    def on_wait(self):
        self.wait_process.start()
        self.network.eval()

    def on_stop(self):
        self.stop_event.set()
        self.memories.need_evaluate.set() # to stop event
        self.wait_process = None

    def save_checkpoint(self):
        if not os.path.exists(SAVE_MODEL_PATH):
            os.makedirs(SAVE_MODEL_PATH)

        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'elo': self.elo
        }, SAVE_MODEL_PATH + MODEL_NAME)

        print('checkpoint saved')

    def load_checkpoint(self):
        if os.path.isfile(SAVE_MODEL_PATH + MODEL_NAME):
            checkpoint = torch.load(SAVE_MODEL_PATH + MODEL_NAME, map_location=self.device)
            self.network.load_state_dict(checkpoint['network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.elo = checkpoint['elo']

            print('checkpoint loaded')

            return True

        return False
