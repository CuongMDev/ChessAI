import numpy as np
from config.EnvConfig import BOARD_SIZE, POLICY_OUT_CHANNEL

class UciMapping:
    __rook_directions = [ (0, 1), (1, 0), (0, -1), (-1, 0) ]
    __bishop_directions = [ (1, 1), (-1, 1), (1, -1), (-1, -1) ]
    __knight_directions = [ (2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1) ]

    __letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    __numbers = ['8', '7', '6', '5', '4', '3', '2', '1']
    __promoted_to = ['q', 'r', 'b', 'n']

    def __init__(self, network_type):
        self.network_type = network_type
        self.labels_array = UciMapping.__create_uci_labels()

        self.mask_index = np.array([self.__uci_to_mask_index(uci) for uci in self.labels_array], dtype=np.int32)
        idx = np.argsort(self.mask_index)

        self.mask_index = self.mask_index[idx]
        self.labels_array = self.labels_array[idx]

        if self.network_type == 'attention':
            self.dict = np.empty(4096 + 32 * 8, dtype=np.int32)
        elif self.network_type == 'cnn':
            self.dict = np.empty(POLICY_OUT_CHANNEL * BOARD_SIZE ** 2, dtype=np.int32)

        for v, k in enumerate(self.mask_index):
            self.dict[k] = v

    def __uci_to_mask_index(self, uci: str) -> int:
        l1 = UciMapping.__letters.index(uci[0])   # file from
        n1 = UciMapping.__numbers.index(uci[1])   # rank from
        l2 = UciMapping.__letters.index(uci[2])   # file to
        n2 = UciMapping.__numbers.index(uci[3])   # rank to

        if self.network_type == 'attention':
            from_index = n1 * BOARD_SIZE + l1
            to_index   = n2 * BOARD_SIZE + l2

            if len(uci) == 5:  # promotion
                p = UciMapping.__promoted_to.index(uci[4])  # q,r,b,n → 0..3
                # mapping khớp với (B, 8, 32)
                # index = 4096 + file * 32 + p*8 + rank_to
                return 4096 + l1 * 32 + p * 8 + l2
            else:
                return from_index * (BOARD_SIZE**2) + to_index
        elif self.network_type == 'cnn':
            if len(uci) == 5:
                p = UciMapping.__promoted_to.index(uci[4])
                f = len(UciMapping.__rook_directions) * 7 + len(UciMapping.__bishop_directions) * 7 + len(
                    UciMapping.__knight_directions) + p * 3 + l2 - l1 + 1
            elif l1 == l2 or n1 == n2:
                if l1 == l2:
                    dis = abs(n2 - n1)
                    dl = 0
                    dr = (n2 - n1) // dis
                else:
                    dis = abs(l2 - l1)
                    dl = (l2 - l1) // dis
                    dr = 0
                direction_index = UciMapping.__rook_directions.index((dl, dr))
                f = direction_index * 7 + (dis - 1)

            elif abs(l2 - l1) == abs(n2 - n1):
                dis = abs(l2 - l1)
                dl = (l2 - l1) // dis
                dr = (n2 - n1) // dis
                direction_index = UciMapping.__bishop_directions.index((dl, dr))
                f = len(UciMapping.__rook_directions) * 7 + direction_index * 7 + (dis - 1)

            else:
                dl = l2 - l1
                dr = n2 - n1
                f = len(UciMapping.__rook_directions) * 7 + len(
                    UciMapping.__bishop_directions) * 7 + UciMapping.__knight_directions.index((dl, dr))

            return f * BOARD_SIZE ** 2 + n1 * BOARD_SIZE + l1

    def __mask_index_to_uci(self, index: int) -> str:
        if self.network_type == 'attention':
            # --- Phong cấp ---
            if index >= 4096:
                temp = index - 4096
                l1 = temp // 32
                remain = temp % 32
                p = remain // 8
                l2 = remain % 8

                # Với phong cấp, mặc định rank 7 -> rank 8
                return f"{UciMapping.__letters[l1]}7{UciMapping.__letters[l2]}8{UciMapping.__promoted_to[p]}"

            # --- Di chuyển bình thường ---
            from_index = index // (BOARD_SIZE ** 2)
            to_index = index % (BOARD_SIZE ** 2)
            n1, l1 = divmod(from_index, BOARD_SIZE)
            n2, l2 = divmod(to_index, BOARD_SIZE)
            return f"{UciMapping.__letters[l1]}{UciMapping.__numbers[n1]}{UciMapping.__letters[l2]}{UciMapping.__numbers[n2]}"

        elif self.network_type == 'cnn':
            rook_dirs = UciMapping.__rook_directions
            bishop_dirs = UciMapping.__bishop_directions
            knight_dirs = UciMapping.__knight_directions

            f, remain = divmod(index, BOARD_SIZE ** 2)
            n1, l1 = divmod(remain, BOARD_SIZE)

            num_rook = len(rook_dirs) * 7
            num_bishop = len(bishop_dirs) * 7
            num_knight = len(knight_dirs)

            # --- Rook move ---
            if f < num_rook:
                direction_index = f // 7
                dis = (f % 7) + 1
                dl, dr = rook_dirs[direction_index]
                l2 = l1 + dl * dis
                n2 = n1 + dr * dis

            # --- Bishop move ---
            elif f < num_rook + num_bishop:
                f -= num_rook
                direction_index = f // 7
                dis = (f % 7) + 1
                dl, dr = bishop_dirs[direction_index]
                l2 = l1 + dl * dis
                n2 = n1 + dr * dis

            # --- Knight move ---
            elif f < num_rook + num_bishop + num_knight:
                f -= num_rook + num_bishop
                dl, dr = knight_dirs[f]
                l2 = l1 + dl
                n2 = n1 + dr

            # --- Promotion ---
            else:
                f -= num_rook + num_bishop + num_knight
                p = f // 3
                shift = f % 3 - 1
                l2 = l1 + shift
                # mặc định rank 7 -> rank 8
                return f"{UciMapping.__letters[l1]}7{UciMapping.__letters[l2]}8{UciMapping.__promoted_to[p]}"

            return f"{UciMapping.__letters[l1]}{UciMapping.__numbers[n1]}{UciMapping.__letters[l2]}{UciMapping.__numbers[n2]}"

    def get_dict_value(self, key):
        return self.dict[self.__uci_to_mask_index(key)]

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