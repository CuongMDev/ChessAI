import os

import chess
import chess.syzygy
import numpy as np
from chess import STARTING_FEN
import chess.pgn

from Env.UciMapping import get_dict_value
from Utils.Utils import Utils
from config.config import LABELS_MAP, TABLEBASE_PATH, BONUS_END_POSITION
from config.EnvConfig import BOARD_SIZE, PIECES_ORDER, INPUT_PIECE_STATES

if not os.path.isdir(TABLEBASE_PATH):
    os.makedirs(TABLEBASE_PATH)
TABLEBASE = chess.syzygy.open_tablebase(TABLEBASE_PATH)

class GameState:
    def __init__(self, pre_env:chess.Board = None, history_state=None, fen=STARTING_FEN):
        if pre_env is None:
            self._env = chess.Board(fen)
        else:
            self._env = pre_env

        if history_state is None:
            self.all_board_states = np.zeros(INPUT_PIECE_STATES, dtype=np.int64)
        else:
            self.all_board_states = history_state

        board_one_hot = self.chess_env_to_one_hot_board()
        self.all_board_states = np.concatenate((self.all_board_states, board_one_hot))[-INPUT_PIECE_STATES:]

        self.is_terminate = False

        self.has_sticky_result = False
        self.can_have_sticky_result = True
        self.result = None

    def ply(self):
        return self._env.ply()

    def get_pgn(self):
        return chess.pgn.Game.from_board(self._env)

    def chess_env_to_one_hot_board(self):
        turn = self._env.turn

        # Chuyển đổi bàn cờ thành mảng 2D, mỗi ô chứa ký tự đại diện quân cờ
        board_one_hot = np.zeros(len(PIECES_ORDER), dtype=np.int64)
        for row in range(BOARD_SIZE):
            # Lấy mỗi hàng từ bàn cờ và chuyển thành mảng con
            for col in range(BOARD_SIZE):
                square = chess.square(col, BOARD_SIZE - 1 - row if turn == chess.WHITE else row)  # Để đảm bảo thứ tự từ dưới lên
                piece = self._env.piece_at(square)
                if piece is not None:
                    piece = PIECES_ORDER.index(piece.symbol() if turn == chess.WHITE else piece.symbol().swapcase())  # Lấy ký tự của quân cờ
                    board_one_hot[piece] |= np.int64(1) << chess.square(col, row)

        if self._env.has_legal_en_passant():
            ep_square = self._env.ep_square
            if turn == chess.WHITE:
                ep_square = self.__flip_rank(ep_square)
            board_one_hot[PIECES_ORDER.index('E')] = np.int64(1) << ep_square

        return board_one_hot

    def is_start_position(self):
        return self._env.fen() == chess.STARTING_FEN

    def get_network_input(self):
        board_2d = np.array(self.all_board_states, dtype=np.int64)

        turn = self._env.turn
        # half_move must be in last
        return np.array([
                *board_2d,
                self._env.has_kingside_castling_rights(turn),
                self._env.has_queenside_castling_rights(turn),
                self._env.has_kingside_castling_rights(not turn),
                self._env.has_queenside_castling_rights(not turn),
                self._env.is_repetition(2),
                self._env.turn,
                min(100, self._env.halfmove_clock)
            ], dtype=np.int64)

    def inverse_history(self):
        inverse_history = np.zeros(INPUT_PIECE_STATES - 1, dtype=np.int64) # not get ep
        for i in range(0, len(self.all_board_states) - 1, len(PIECES_ORDER) - 1): # not get ep
            for j in range(len(PIECES_ORDER) - 1):
                piece_j = PIECES_ORDER[j]
                inverse_history[i + j] = Utils.flip_rows(self.all_board_states[i + PIECES_ORDER.index(piece_j.swapcase())])

        return inverse_history

    def rollback(self):
        previous_state = GameState(self._env.copy(stack=True))
        previous_state._env.pop()
        return previous_state

    @staticmethod
    def __flip_rank(square):
        file = chess.square_file(square)
        rank = BOARD_SIZE - 1 - chess.square_rank(square)
        return chess.square(file, rank)

    @staticmethod
    def __flip_move_vertically(chess_move):
        return chess.Move(
            from_square=GameState.__flip_rank(chess_move.from_square),
            to_square=GameState.__flip_rank(chess_move.to_square),
            promotion=chess_move.promotion
        )

    def get_legal_moves(self):
        legal_chess_moves = self._env.legal_moves
        turn = self._env.turn
        legal_moves = [get_dict_value(LABELS_MAP.dict, (chess_move if turn == chess.WHITE else self.__flip_move_vertically(chess_move)).uci()) for chess_move in legal_chess_moves]

        return legal_moves

    def get_last_real_uci(self, last_move):
        # with parent view

        parent_turn = not self._env.turn
        move_uci = LABELS_MAP.labels_array[last_move]
        chess_move = chess.Move.from_uci(move_uci)
        if parent_turn != chess.WHITE:
            chess_move = self.__flip_move_vertically(chess_move)
        return chess_move.uci()

    def real_uci_to_move(self, move_uci):
        turn = self._env.turn
        chess_move = chess.Move.from_uci(move_uci)
        if turn != chess.WHITE:
            chess_move = self.__flip_move_vertically(chess_move)
        return get_dict_value(LABELS_MAP.dict, chess_move.uci())

    def perform_move(self, move, copy_full_stack=False, claim_draw=False):
        new_env = self._env.copy(stack=True if copy_full_stack else self._env.halfmove_clock + 1)

        turn = self._env.turn
        move_uci = LABELS_MAP.labels_array[move]
        chess_move = chess.Move.from_uci(move_uci)
        if turn != chess.WHITE:
            chess_move = self.__flip_move_vertically(chess_move)
        new_env.push(chess_move)

        new_state = GameState(new_env, history_state=self.inverse_history())
        wdl = TABLEBASE.get_wdl(new_state._env)
        if wdl is not None:
            if new_state._env.halfmove_clock + abs(TABLEBASE.probe_dtz(new_state._env)) < 100:
                wdl = wdl // 2
                new_state.result = wdl
                if wdl == 1:
                    claim_draw = False # can win
            else:
                new_state.result = 0
            new_state.has_sticky_result = True

        result = new_state._env.result(claim_draw=claim_draw)
        if result == '1/2-1/2':
            new_state.result = 0
            new_state.has_sticky_result = True
            new_state.is_terminate = True
        elif result != '*':
            new_state.result = -1 # end game -> next state turn lose
            new_state.has_sticky_result = True
            new_state.is_terminate = True

        return new_state

    def score(self):
        if self.is_terminate:
            return self.result + np.sign(self.result) * BONUS_END_POSITION
        return self.result