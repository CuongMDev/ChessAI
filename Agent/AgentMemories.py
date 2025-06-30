from multiprocessing import Event, Value
import torch

from config.config import BOARD_SIZE, INFO_SIZE, NUM_WORKERS, MIN_EVALUATE_COUNT, LABELS_MAP


class AgentMemories:
    def __init__(self, pre_data=None, num_workers=NUM_WORKERS, min_evaluate_count=MIN_EVALUATE_COUNT):
        # shared memory
        if pre_data is None:
            self.share_input = torch.zeros(num_workers, BOARD_SIZE, BOARD_SIZE + INFO_SIZE, dtype=torch.int32).share_memory_() # -1 thông tin ep
            self.share_input_mask = torch.zeros(num_workers, 512, dtype=torch.int32).share_memory_()
            self.share_input_mask_size = torch.zeros(num_workers, dtype=torch.int32).share_memory_()

            self.share_output_pol = torch.zeros(num_workers, 512).share_memory_()
            self.share_output_val = torch.zeros(num_workers, 3).share_memory_()
            self.share_valid_data = [Event() for _ in range(num_workers)]
            for event in self.share_valid_data:
                event.set()

            self.need_evaluate_count = Value('i', 0)
            self.need_evaluate = Event()
            self.num_current_worker = Value('i', num_workers)
        else:
            (self.share_input,
            self.share_input_mask,
            self.share_input_mask_size,
            self.share_output_pol,
            self.share_output_val,
            self.share_valid_data,
            self.need_evaluate_count,
            self.need_evaluate,
            self.num_current_worker) = pre_data

        self.num_workers = num_workers
        self.min_evaluate_count = min_evaluate_count

    def get_share_memory(self):
        return (self.share_input,
                self.share_input_mask,
                self.share_input_mask_size,
                self.share_output_pol,
                self.share_output_val,
                self.share_valid_data,
                self.need_evaluate_count,
                self.need_evaluate,
                self.num_current_worker)

    def get_evaluation(self, i):
        return self.share_output_pol[i, :self.share_input_mask_size[i]], self.share_output_val[i]

    def add_evaluation_req(self, i, inp, mask):
        mask_len = len(mask)
        self.share_input[i] = inp
        self.share_input_mask[i, :mask_len] = torch.tensor(mask)
        self.share_input_mask_size[i] = mask_len

        self.share_valid_data[i].clear()
        self.change_need_predict_count(1)

    def change_need_predict_count(self, value):
        with self.need_evaluate_count.get_lock(), self.num_current_worker.get_lock():
            self.need_evaluate_count.value += value

            if self.num_current_worker.value != 0 and self.need_evaluate_count.value >= min(self.num_current_worker.value, self.min_evaluate_count):
                self.need_evaluate.set()

    def change_current_worker_count(self, value):
        with self.need_evaluate_count.get_lock(), self.num_current_worker.get_lock():
            self.num_current_worker.value += value

            if self.num_current_worker.value != 0 and self.need_evaluate_count.value >= min(self.num_current_worker.value, self.min_evaluate_count):
                self.need_evaluate.set()

    def clear(self):
        self.num_current_worker.value = 0

    def reset(self):
        self.num_current_worker.value = self.num_workers