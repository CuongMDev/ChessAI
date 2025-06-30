import numpy as np
from numba.typed import Dict
from numba import types, njit, int32

BOARD_SIZE = 8
POLICY_OUT_CHANNEL = BOARD_SIZE * BOARD_SIZE + 12

@njit
def get_dict_value(dict, key):
    return dict[key]

class UciMapping:
    __letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    __numbers = ['1', '2', '3', '4', '5', '6', '7', '8']
    __promoted_to = ['q', 'r', 'b', 'n']

    def __init__(self):
        self.labels_array = UciMapping.__create_uci_labels()

        key = np.array([UciMapping.__uci_to_mask_index(uci) for uci in self.labels_array], dtype=np.int32)
        idx = np.lexsort((key[:, 2], key[:, 1], key[:, 0]))

        self.labels_array = self.labels_array[idx]
        self.dict = Dict.empty(
            key_type=types.unicode_type,
            value_type=types.int32
        )
        for v, k in enumerate(self.labels_array):
            self.dict[k] = int32(v)

    @staticmethod
    def __uci_to_mask_index(uci):
        l1 = UciMapping.__letters.index(uci[0])
        n1 = UciMapping.__numbers.index(uci[1])
        l2 = UciMapping.__letters.index(uci[2])
        n2 = UciMapping.__numbers.index(uci[3])
        if len(uci) == 5:
            p = UciMapping.__promoted_to.index(uci[4])
            n = BOARD_SIZE * BOARD_SIZE + p * 3 + l2 - l1 + 1
        else:
            n = l2 * BOARD_SIZE + n2

        return n, l1, n1

    @staticmethod
    def __create_uci_labels():
        """
        Creates the labels for the universal chess interface into an array and returns them
        :return:
        """
        labels_array = []

        for l1 in range(8):
            for n1 in range(8):
                destinations = [(t, n1) for t in range(8)] + \
                               [(l1, t) for t in range(8)] + \
                               [(l1 + t, n1 + t) for t in range(-7, 8)] + \
                               [(l1 + t, n1 - t) for t in range(-7, 8)] + \
                               [(l1 + a, n1 + b) for (a, b) in
                                [(-2, -1), (-1, -2), (-2, 1), (1, -2), (2, -1), (-1, 2), (2, 1), (1, 2)]]
                for (l2, n2) in destinations:
                    if (l1, n1) != (l2, n2) and l2 in range(8) and n2 in range(8):
                        move = UciMapping.__letters[l1] + UciMapping.__numbers[n1] + UciMapping.__letters[l2] + UciMapping.__numbers[n2]
                        labels_array.append(move)
        for l1 in range(8):
            l = UciMapping.__letters[l1]
            for p in UciMapping.__promoted_to:
                labels_array.append(l + '7' + l + '8' + p)
                if l1 > 0:
                    l_l = UciMapping.__letters[l1 - 1]
                    labels_array.append(l + '7' + l_l + '8' + p)
                if l1 < 7:
                    l_r = UciMapping.__letters[l1 + 1]
                    labels_array.append(l + '7' + l_r + '8' + p)

        return np.array(labels_array, dtype='U5')

    def create_uci_labels_mask(self):
        labels_mask = np.zeros((POLICY_OUT_CHANNEL, BOARD_SIZE, BOARD_SIZE), dtype=bool)
        for uci in self.labels_array:
            labels_mask[UciMapping.__uci_to_mask_index(uci)] = True
        return labels_mask