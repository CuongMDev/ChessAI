import numpy as np
import torch

from Agent.AgentMemories import AgentMemories
from Utils.Utils import compute_kld
from config.ConfigManager import ConfigManager
from Env.GameState import GameState
from MonteCarloTreeSearch.MonteCarloNode import MonteCarloNode
from config.config import DIRICHLET_ALPHA, DIRICHLET_EPSILON, LABELS_MAP, MAX_THINK_LOOP, \
    KLD_THRESHOLD, FPU_VALUE_AT_ROOT, TEMPERATURE_VISIT_OFFSET, SMART_PRUNING_FACTOR, ROOT_EXPLORATION_WEIGHT


class MonteCarloTreeSearch:
    def __init__(self, agent_memories: AgentMemories, config: ConfigManager, worker=0, fen=None, is_training=True, auto_claim_draw=False):
        self.agent_memories = agent_memories
        self.worker = worker
        self.auto_claim_draw = auto_claim_draw

        self.config = config
        self.is_training = is_training

        if fen is None:
            self.root = MonteCarloNode(GameState())
        else:
            self.root = MonteCarloNode(GameState(fen=fen))

        self.is_start_position = self.root.state.is_start_position()
        self.expand_first_root()

    def search(self, temperature):
        self.boost_root_dtz()

        for loop in range(MAX_THINK_LOOP):
            for simulation in range(self.config.NUM_SIMULATION):
                leaf = self.traverse(self.config.NUM_SIMULATION - simulation)
                simulation_result = self.rollout(leaf)
                leaf.backpropagate(simulation_result)

            best_child, pi = self.choose_child(self.root, temperature, check_kld=not loop==MAX_THINK_LOOP - 1)
            if best_child is not None:
                return best_child, pi

    def update_mcts_root_from_move(self, move):
        if self.root.children:
            for child in self.root.children:
                if child.last_move == move:
                    return self.update_mcts_root(child)

        self.update_mcts_root(self.root.expand_specific(move, 0))

    def update_mcts_root(self, move_node):
        """Cập nhật cây MCTS sau khi chọn nước đi."""
        if move_node is None:
            raise ValueError("Không thể cập nhật root vì move_node là None")

        # print(-self.root.children_info.average_values[move_node.id], flush=True)
        # print(max(-self.root.children_info.average_values), flush=True)

        self.is_start_position = False

        # Node được chọn trở thành root mới
        self.root = move_node
        self.root.get_state(copy_full_stack=True, claim_draw=self.auto_claim_draw)

        # Giữ lại các node con của root mới
        self.root.parent = None  # Xóa liên kết đến node cha cũ để tiết kiệm bộ nhớ

    def rollback(self):
        self.root = MonteCarloNode(self.root.state.rollback())
        self.root.visit = 1

    def can_overtake_bestmove(self, node, remaining_visits):
        """
        Trả về True nếu current move còn cơ hội vượt best move,
        tức là sẽ không bị prune.
        """
        # Giả định nước yếu được dồn toàn bộ lượt còn lại
        max_possible_visits = node.visit + remaining_visits / SMART_PRUNING_FACTOR

        return max_possible_visits > self.root.best_child_visit

    def boost_root_dtz(self):
        self.expand(self.root)
        if self.root.state.result_tablebase is not None:
            self.root.children_info.priors *= self.root.state.get_tablebase_policy()
            self.root.children_info.priors /= np.sum(self.root.children_info.priors)

    def expand_first_root(self):
        policies, _ = self.get_evaluation(self.root, self.root.state.get_legal_moves())
        if self.is_training:
            # add dirichlet noise
            dir_noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(policies)).astype(np.float32)
            policies = (1 - DIRICHLET_EPSILON) * policies + DIRICHLET_EPSILON * dir_noise

        self.root.visit = 1
        self.root.expand(policies, FPU_VALUE_AT_ROOT)

    def get_evaluation(self, node, legal_move):
        state_tensor = torch.from_numpy(node.state.get_train_board())
        self.agent_memories.add_evaluation_req(self.worker, state_tensor, legal_move)
        self.agent_memories.share_valid_data[self.worker].wait()

        policies, value = self.agent_memories.get_evaluation(self.worker)

        policies /= self.config.POLICY_SOFTMAX_TEMP
        policies = torch.softmax(policies, dim=-1).numpy()

        value = torch.softmax(value, dim=-1)
        value = (value[2] - value[0]).item()

        return policies, value

    def traverse(self, remaining_visits) -> MonteCarloNode:
        node = self.root
        if node.is_fully_expanded:
            while True:
                node = self.root.best_child(ROOT_EXPLORATION_WEIGHT if self.is_start_position else self.config.EXPLORATION_WEIGHT)
                if not self.is_training and not self.can_overtake_bestmove(node, remaining_visits):
                    self.root.children_info.average_values[node.id] = 1e9  # no visit anymore
                    continue  # continue find new node
                break

        while node.is_fully_expanded and node.children:
            node = node.best_child(self.config.EXPLORATION_WEIGHT)
            if node.state is not None and node.state.result_tablebase is not None:
                break # break avoid wrong policy

        return node

    def expand(self, node):
        if not node.is_fully_expanded and not node.state.is_terminate:
            policies, value = self.get_evaluation(node, node.state.get_legal_moves())
            node.expand(policies, max(-1, value - self.config.FPU_VALUE))

            return value

        return None

    def rollout(self, node):
        node.get_state(copy_full_stack=False, claim_draw=True) # AI auto claim draw
        value = self.expand(node)

        score = node.state.score()
        if score is not None:
            return score

        return value

    def choose_child(self, node, temperature, check_kld=False):
        visits = node.children_info.visits
        # if check_kld and compute_kld(self.root.children_info.priors, visits / sum(visits)) > KLD_THRESHOLD:
        #     return None, None

        if self.is_training:
            pi = np.zeros(len(LABELS_MAP.labels_array))
            for i, child in enumerate(node.children):
                pi[child.last_move] = visits[i]
            pi /= np.sum(pi)

            for child in node.children:
                pi[child.last_move] += 1 # Lưu cả thông tin mask, > 0 => mask = true
        else:
            pi = None

        if temperature == 0:
            max_visit = np.max(visits)
            best_indices = np.flatnonzero(visits == max_visit)
            chosen_index = np.random.choice(best_indices)

            return node.children[chosen_index], pi

        probs = (visits - TEMPERATURE_VISIT_OFFSET) ** (1 / temperature)
        probs /= np.sum(probs)

        return np.random.choice(node.children, p=probs), pi