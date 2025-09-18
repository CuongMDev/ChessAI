from typing import Optional

import numpy as np
import scipy

from config.EnvConfig import FULL_INPUT_STATES
from MonteCarloTreeSearch import MonteCarloNode
from MonteCarloTreeSearch._MonteCarloTreeSearch import _MonteCarloTreeSearch
from config.ConfigManager import ConfigManager
from config.config import PLAY_THREAD


class MonteCarloTreeSearchPlay(_MonteCarloTreeSearch):
    def __init__(self, session, config: ConfigManager, worker=0, fen=None, is_training=True, auto_claim_draw=False):
        self.session = session
        self.input_name = session.get_inputs()[0].name

        self.play_thread = PLAY_THREAD

        self.need_evaluate_node: list[Optional[MonteCarloNode]] = [None] * self.play_thread
        self.is_traversing_node: list[Optional[MonteCarloNode]] = [None] * self.play_thread
        self.current_evaluate_pos = 0
        self.current_traversing_pos = 0

        super().__init__(config, worker, fen, is_training, auto_claim_draw)

    def set_play_thread(self, search_thread):
        self.play_thread = search_thread
        self.need_evaluate_node: list[Optional[MonteCarloNode]] = [None] * self.play_thread
        self.is_traversing_node: list[Optional[MonteCarloNode]] = [None] * self.play_thread

    def add_traversing(self, left_simulation, use_smart_pruning, start_node: MonteCarloNode = None):
        if start_node is None:
            node = self.root
        else:
            node = start_node
        node.is_evaluating = False

        leaf = self.traverse(node, left_simulation, use_smart_pruning=use_smart_pruning)

        if leaf.is_evaluating:
            leaf.backpropagate_virtual_loss(remove_virtual_loss=False, stop_node=start_node)
            self.is_traversing_node[self.current_traversing_pos] = leaf
            self.current_traversing_pos += 1
        else:
            leaf.get_state(copy_full_stack=False, claim_draw=True)  # AI auto claim draw

            leaf.is_evaluating = True
            leaf.backpropagate_virtual_loss(remove_virtual_loss=False, stop_node=start_node)
            self.need_evaluate_node[self.current_evaluate_pos] = leaf
            self.current_evaluate_pos += 1

    def search(self, temperature):
        if not self.root.is_fully_expanded:
            value = self.rollout(self.root, force_expand=True)
            self.root.backpropagate(value)

        use_smart_pruning = temperature == 0

        num_simulation = self.config.NUM_SIMULATION
        while num_simulation + self.current_traversing_pos > 0:
            traversing_num = self.current_traversing_pos
            self.current_traversing_pos = 0
            self.current_evaluate_pos = 0
            for i in range(traversing_num):
                self.add_traversing(num_simulation + traversing_num - i, use_smart_pruning=use_smart_pruning, start_node=self.is_traversing_node[i])

            new_traverse_num = min(self.play_thread - self.current_evaluate_pos - self.current_traversing_pos, num_simulation)

            stop_early=False
            for i in range(new_traverse_num):
                self.add_traversing(num_simulation, use_smart_pruning=use_smart_pruning)
                num_simulation -= 1
                if use_smart_pruning and self.can_stop_early(num_simulation):
                    stop_early=True
                    break

            policies, values = self.get_all_evaluation()
            need_evaluate_network_num = 0
            for i in range(self.current_evaluate_pos):
                node = self.need_evaluate_node[i]
                if node.state.has_sticky_result:
                    value_i = node.state.score()
                else:
                    legal_moves = node.state.get_legal_moves()
                    policy_i = policies[need_evaluate_network_num, legal_moves]
                    value_i = values[need_evaluate_network_num]
                    need_evaluate_network_num += 1

                    policy_i /= self.config.POLICY_SOFTMAX_TEMP
                    policy_i = scipy.special.softmax(policy_i, axis=-1)

                    value_i = scipy.special.softmax(value_i, axis=-1)
                    value_i = value_i[2] - value_i[0]

                    node.expand(policy_i, max(-1, value_i - self.config.FPU_VALUE))

                node.backpropagate_virtual_loss(remove_virtual_loss=True)
                node.backpropagate(value_i)

            if stop_early:
                # clean
                for i in range(self.current_traversing_pos):
                    self.is_traversing_node[i].backpropagate_virtual_loss(remove_virtual_loss=True)
                self.current_traversing_pos = 0
                break

        best_child, pi = self.choose_child(self.root, temperature)
        if best_child is not None:
            return best_child, pi

    def get_all_evaluation(self):
        need_evaluate_network = [i for i in range(self.current_evaluate_pos) if not self.need_evaluate_node[i].state.has_sticky_result]
        all_states = np.empty((len(need_evaluate_network), FULL_INPUT_STATES), dtype=np.int64)
        for i, need_evaluate_network_i in enumerate(need_evaluate_network):
            all_states[i] = self.need_evaluate_node[need_evaluate_network_i].state.get_network_input()

        policies, values = self.session.run(None, {self.input_name: all_states})
        return policies, values

    def get_evaluation(self, node, legal_move):
        state = node.state.get_network_input()[np.newaxis, :]

        policy, value = self.session.run(None, {self.input_name: state})
        policy = policy.squeeze()[legal_move]
        value = value.squeeze()

        policy /= self.config.POLICY_SOFTMAX_TEMP
        policy = scipy.special.softmax(policy, axis=-1)

        value = scipy.special.softmax(value, axis=-1)
        value = value[2] - value[0]

        return policy, value