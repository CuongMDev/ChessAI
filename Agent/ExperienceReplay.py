from multiprocessing import Value
import torch
from torch.utils.data import DataLoader, Dataset, random_split, Subset

from config.NetworkConfig import INFO_SIZE, DEVICE
from config.config import BOARD_SIZE, EXP_MAX, LABELS_MAP


class ExperienceReplay(Dataset):
    def __init__(self, pre_data=None):
        if pre_data is None:
            self.memory_state = torch.empty(EXP_MAX, BOARD_SIZE, BOARD_SIZE + INFO_SIZE, dtype=torch.int32).share_memory_()
            self.memory_pi = torch.empty(EXP_MAX, len(LABELS_MAP.labels_array)).share_memory_()
            self.memory_reward = torch.empty(EXP_MAX, dtype=torch.long).share_memory_()
            self.game_index = torch.empty(EXP_MAX, dtype=torch.long).share_memory_()
            self.game_num = Value('i', 0)
            self.index = Value('i', 0) # index thêm data vào
            self.size = Value('i', 0) #  số data trong memory đã được lấp đầy
        else:
            (self.memory_state,
            self.memory_pi,
            self.memory_reward,
            self.game_index,
            self.game_num,
            self.index,
            self.size) = pre_data

        self.memory_reward_device = None
        self.memory_pi_device = None
        self.memory_state_device = None

    def get_share_memory(self):
        return self.memory_state, self.memory_pi, self.memory_reward, self.game_index, self.game_num, self.index, self.size

    def create_device_memory(self):
        self.memory_state_device = self.memory_state[: self.size.value].to(DEVICE)
        self.memory_pi_device = self.memory_pi[: self.size.value].to(DEVICE)
        self.memory_reward_device = self.memory_reward[: self.size.value].to(DEVICE)

    def reset(self):
        self.index.value = 0
        self.size.value = 0
        self.game_num.value = 0

    def delete_device_memory(self):
        self.memory_state_device = None
        self.memory_pi_device = None
        self.memory_reward_device = None

    def add_experience(self, sample: list):
        if len(sample) != 3:
            raise Exception('Invalid sample')
        current_index = self.index.value
        self.index.value = (self.index.value + 1) % EXP_MAX
        self.size.value = min(self.size.value + 1, EXP_MAX)

        self.memory_state[current_index] = torch.from_numpy(sample[0])
        self.memory_pi[current_index] = torch.from_numpy(sample[1])
        self.memory_reward[current_index] = sample[2]
        self.game_index[current_index] = self.game_num.value

    def add_experiences(self, samples: list):
        with self.index.get_lock(), self.size.get_lock(), self.game_num.get_lock():
            for sample in samples:
                self.add_experience(sample)

            self.game_num.value += 1

    def sample_experience(self, sample_size: int):
        indices = torch.randperm(self.size.value, device=DEVICE)[:sample_size]  # Lấy sample_size chỉ số ngẫu nhiên từ 0 đến n-1

        # Tách dữ liệu
        state_batch = self.memory_state_device[indices]
        pi_batch = self.memory_pi_device[indices]
        reward_batch = self.memory_reward_device[indices]

        return state_batch, pi_batch, reward_batch

    def get_all_data(self, batch_size, validation_split=0):
        self.create_device_memory()

        all_game_ids = torch.unique(self.game_index)

        # Bước 2: Shuffle danh sách game IDs nếu cần
        shuffled_ids = all_game_ids[torch.randperm(len(all_game_ids))]

        # Bước 3: Chia theo tỉ lệ
        val_size = int(validation_split * len(shuffled_ids))
        train_ids = shuffled_ids[val_size:]
        val_ids = shuffled_ids[:val_size]

        # Bước 4: Lấy index của positions thuộc từng tập
        train_indices = (self.game_index[: self.size.value].unsqueeze(1) == train_ids).any(dim=1).nonzero(as_tuple=True)[0]
        val_indices = (self.game_index[: self.size.value].unsqueeze(1) == val_ids).any(dim=1).nonzero(as_tuple=True)[0]

        train_dataset = Subset(self, train_indices)
        val_dataset = Subset(self, val_indices)

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
