from multiprocessing import Value
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from config.config import BOARD_SIZE, DEVICE, EXP_MAX, INFO_SIZE, LABELS_MAP


class ExperienceReplay(Dataset):
    def __init__(self, pre_data=None):
        if pre_data is None:
            self.memory_state = torch.empty(EXP_MAX, BOARD_SIZE, BOARD_SIZE + INFO_SIZE, dtype=torch.int32).share_memory_()
            self.memory_pi = torch.empty(EXP_MAX, len(LABELS_MAP.labels_array)).share_memory_()
            self.memory_reward = torch.empty(EXP_MAX, dtype=torch.long).share_memory_()
            self.index = Value('i', 0) # index thêm data vào
            self.size = Value('i', 0) #  số data trong memory đã được lấp đầy
        else:
            self.memory_state = pre_data[0]
            self.memory_pi = pre_data[1]
            self.memory_reward = pre_data[2]
            self.index = pre_data[3]
            self.size = pre_data[4]

        self.memory_reward_device = None
        self.memory_pi_device = None
        self.memory_state_device = None

    def get_share_memory(self):
        return self.memory_state, self.memory_pi, self.memory_reward, self.index, self.size

    def create_device_memory(self):
        self.memory_state_device = self.memory_state[: self.size.value].to(DEVICE)
        self.memory_pi_device = self.memory_pi[: self.size.value].to(DEVICE)
        self.memory_reward_device = self.memory_reward[: self.size.value].to(DEVICE)

    def reset(self):
        self.index.value = 0
        self.size.value = 0

    def delete_device_memory(self):
        self.memory_state_device = None
        self.memory_pi_device = None
        self.memory_reward_device = None

    def add_experience(self, sample: list):
        if len(sample) != 3:
            raise Exception('Invalid sample')
        with self.index.get_lock(), self.size.get_lock():
            current_index = self.index.value
            self.index.value = (self.index.value + 1) % EXP_MAX
            self.size.value = min(self.size.value + 1, EXP_MAX)

        self.memory_state[current_index] = torch.from_numpy(sample[0])
        self.memory_pi[current_index] = torch.from_numpy(sample[1])
        self.memory_reward[current_index] = sample[2]

    def sample_experience(self, sample_size: int):
        indices = torch.randperm(self.size.value, device=DEVICE)[:sample_size]  # Lấy sample_size chỉ số ngẫu nhiên từ 0 đến n-1

        # Tách dữ liệu
        state_batch = self.memory_state_device[indices]
        pi_batch = self.memory_pi_device[indices]
        reward_batch = self.memory_reward_device[indices]

        return state_batch, pi_batch, reward_batch

    def get_all_data(self, batch_size, validation_split=0):
        self.create_device_memory()

        val_size = int(validation_split * len(self))
        train_size = len(self) - val_size
        train_dataset, val_dataset = random_split(self, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        if val_size == 0:
            val_loader = None
        else:
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    def __len__(self):
        return self.size.value

    def __getitem__(self, idx):
        return self.memory_state_device[idx], self.memory_pi_device[idx], self.memory_reward_device[idx]
