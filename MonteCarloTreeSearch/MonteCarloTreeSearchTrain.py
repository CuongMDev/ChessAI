import torch

from Agent.AgentMemories import AgentMemories
from MonteCarloTreeSearch._MonteCarloTreeSearch import _MonteCarloTreeSearch
from config.ConfigManager import ConfigManager


class MonteCarloTreeSearchTrain(_MonteCarloTreeSearch):
    def __init__(self, agent_memories: AgentMemories, config: ConfigManager, worker=0, fen=None, is_training=True,
                 auto_claim_draw=False):
        self.agent_memories = agent_memories
        super().__init__(config, worker, fen, is_training, auto_claim_draw)\

    def get_evaluation(self, node, legal_move):
        state_tensor = torch.from_numpy(node.state.get_train_input())
        self.agent_memories.add_evaluation_req(self.worker, state_tensor, legal_move)
        self.agent_memories.share_valid_data[self.worker].wait()

        policies, value = self.agent_memories.get_evaluation(self.worker)

        policies /= self.config.POLICY_SOFTMAX_TEMP
        policies = torch.softmax(policies, dim=-1).numpy()

        value = torch.softmax(value, dim=-1)
        value = (value[2] - value[0]).item()

        return policies, value