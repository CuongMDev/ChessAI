import numpy as np
from numba import njit

@njit
def max_puct_score_index_njit(average_values, visits, priors, sqrt_parent_visit, exploration_weight):
    return np.argmax(-average_values + exploration_weight * priors * sqrt_parent_visit / (visits + 1))

class NodeInformation:
    def __init__(self, priors, init_value):
        self.visits = np.zeros(len(priors), dtype=np.int32)
        self.values = np.zeros(len(priors), dtype=np.float32)
        self.average_values = np.full(len(priors), fill_value=-init_value, dtype=np.float32)
        self.priors = priors

        self.has_sticky_endgame = np.zeros(len(priors), dtype=bool)
        self.has_sticky_num = 0

    def add(self, id, reward):
        self.visits[id] += 1
        self.values[id] += reward
        self.average_values[id] = self.values[id] / self.visits[id]

    def add_sticky_result(self, id):
        if not self.has_sticky_endgame[id]:
            self.has_sticky_endgame[id] = True
            self.has_sticky_num += 1

    def has_sticky_result(self):
        return len(self.has_sticky_endgame) == self.has_sticky_num

    def max_puct_score_index(self, sqrt_parent_visit, exploration_weight):
        return max_puct_score_index_njit(self.average_values, self.visits, self.priors, sqrt_parent_visit, exploration_weight)