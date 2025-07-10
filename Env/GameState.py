import numpy as np
import chess
import chess.syzygy
from chess import STARTING_FEN

from Env.UciMapping import get_dict_value
from config.config import PIECES_ORDER, BOARD_SIZE, LABELS_MAP, TABLEBASE_PATH

TABLEBASE = chess.syzygy.open_tablebase(TABLEBASE_PATH)

class GameState:
    def __init__(self, pre_env:chess.Board = None, fen=STARTING_FEN):
        if pre_env is None:
            self._env = chess.Board(fen)
        else:
            self._env = pre_env

        self.is_terminate = False

        self.has_sticky_result = False
        self.can_have_sticky_result = True
        self.result = None

    def ply(self):
        return self._env.ply()

    def chess_env_to_2d_board(self):
        turn = self._env.turn

        # Chuyển đổi bàn cờ thành mảng 2D, mỗi ô chứa ký tự đại diện quân cờ
        board_2d = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int32)
        for row in range(BOARD_SIZE):
            # Lấy mỗi hàng từ bàn cờ và chuyển thành mảng con
            for col in range(BOARD_SIZE):
                square = chess.square(col, BOARD_SIZE - 1 - row if turn == chess.WHITE else row)  # Để đảm bảo thứ tự từ dưới lên
                piece = self._env.piece_at(square)
                if piece is not None:
                    board_2d[row, col] = PIECES_ORDER.index(piece.symbol() if turn == chess.WHITE else piece.symbol().swapcase())  # Lấy ký tự của quân cờ

        return board_2d

    def is_start_position(self):
        return self._env.fen() == chess.STARTING_FEN

    def get_train_input(self):
        board_2d = self.chess_env_to_2d_board()

        turn = self._env.turn
        # ep info
        if self._env.has_legal_en_passant():
            ep_row, ep_col = divmod(self._env.ep_square, BOARD_SIZE)
            if turn == chess.WHITE:
                ep_row = BOARD_SIZE - 1 - ep_row
            board_2d[ep_row, ep_col] = PIECES_ORDER.index('E')

        # info
        turn_can_castle_king = self._env.has_kingside_castling_rights(turn)
        turn_can_castle_queen = self._env.has_queenside_castling_rights(turn)
        reverse_turn_can_castle_king = self._env.has_kingside_castling_rights(not turn)
        reverse_turn_can_castle_queen = self._env.has_queenside_castling_rights(not turn)
        is_repetition = self._env.is_repetition(2)
        half_move = self._env.halfmove_clock

        return np.concatenate([
                    board_2d,
                    np.full((BOARD_SIZE, 1), turn_can_castle_king, dtype=np.int32),
                    np.full((BOARD_SIZE, 1), turn_can_castle_queen, dtype=np.int32),
                    np.full((BOARD_SIZE, 1), reverse_turn_can_castle_king, dtype=np.int32),
                    np.full((BOARD_SIZE, 1), reverse_turn_can_castle_queen, dtype=np.int32),
                    np.full((BOARD_SIZE, 1), is_repetition, dtype=np.int32),
                    np.full((BOARD_SIZE, 1), min(half_move, 100), dtype=np.int32),
        ], axis=1)

    def rollback(self):
        previous_state = GameState(self._env.copy(stack=True))
        previous_state._env.pop()
        return previous_state

    def __flip_rank(self, square):
        file = chess.square_file(square)
        rank = BOARD_SIZE - 1 - chess.square_rank(square)
        return chess.square(file, rank)

    def __flip_move_vertically(self, chess_move):
        return chess.Move(
            from_square=self.__flip_rank(chess_move.from_square),
            to_square=self.__flip_rank(chess_move.to_square),
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
        if copy_full_stack:
            new_state = GameState(self._env.copy(stack=True))
        else:
            new_state = GameState(self._env.copy(stack=self._env.halfmove_clock + 1))

        turn = self._env.turn
        move_uci = LABELS_MAP.labels_array[move]
        chess_move = chess.Move.from_uci(move_uci)
        if turn != chess.WHITE:
            chess_move = self.__flip_move_vertically(chess_move)

        new_state._env.push(chess_move)
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
        return self.result