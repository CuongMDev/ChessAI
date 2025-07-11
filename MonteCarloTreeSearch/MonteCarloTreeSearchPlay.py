import numpy as np
import scipy

from MonteCarloTreeSearch._MonteCarloTreeSearch import _MonteCarloTreeSearch
from config.ConfigManager import ConfigManager


class MonteCarloTreeSearchPlay(_MonteCarloTreeSearch):
    def __init__(self, session, config: ConfigManager, worker=0, fen=None, is_training=True, auto_claim_draw=False):
        self.session = session
        self.input_name = session.get_inputs()[0].name
        super().__init__(config, worker, fen, is_training, auto_claim_draw)

    def get_evaluation(self, node, legal_move):
        state = node.state.get_train_input()[np.newaxis, :]

        policies, value = self.session.run(None, {self.input_name: state})
        policies = policies.squeeze()[legal_move]
        value = value.squeeze()

        policies /= self.config.POLICY_SOFTMAX_TEMP
        policies = scipy.special.softmax(policies, axis=-1)

        value = scipy.special.softmax(value, axis=-1)
        value = value[2] - value[0]

        return policies, value