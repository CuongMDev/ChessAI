from Agent.Agent import Agent
from MonteCarloTreeSearch.MonteCarloTreeSearch import MonteCarloTreeSearch
from config.ConfigManager import ConfigManager
from config.config import DEVICE, TEMPERATURE


class GamePlay:
    def __init__(self):
        self.agent = Agent(DEVICE, 1, 1)
        self.agent.load_checkpoint()
        self.agent.on_wait()
        self.agent.set_jit_mode('trace')

        self.temperature = None
        self.step = None
        self.mcts = None
        self.claimed_draw = None

        self.config = ConfigManager()
        self.config.set_mode('play')

        self.reset()

    def reset(self, fen=None):
        self.mcts = MonteCarloTreeSearch(self.agent.memories, self.config, fen=fen, is_training=False)
        self.step = 0
        self.claimed_draw = False
        self.temperature = TEMPERATURE

    def play(self, move_uci):
        move = self.mcts.root.state.real_uci_to_move(move_uci)
        self.mcts.update_mcts_root_from_move(move)

        return self.mcts.root.state.is_terminate

    def rollback(self):
        self.mcts.rollback()

    def can_claim_draw(self):
        return self.mcts.root.state._env.can_claim_draw()

    def result(self):
        return self.mcts.root.state._env.result(claim_draw=self.claimed_draw)

    # Return move uci
    def ai_play(self):
        if self.can_claim_draw():
            result_tablebase = self.mcts.root.state.result_tablebase
            if result_tablebase is None or result_tablebase != 1:
                self.claimed_draw = True
                return None, True # Draw

        best_move = self.mcts.search(self.temperature)[0]
        self.mcts.update_mcts_root(best_move)

        self.step += 1
        if self.step == self.config.TEMPERATURE_CUTOFF:
            self.temperature = 0

        return self.mcts.root.state.get_last_real_uci(self.mcts.root.last_move), self.mcts.root.state.is_terminate
