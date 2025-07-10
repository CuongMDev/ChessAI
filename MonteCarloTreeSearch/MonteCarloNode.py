import math

from Env.GameState import GameState
from MonteCarloTreeSearch.NodeInformation import NodeInformation


class MonteCarloNode:
    def __init__(self, state: GameState = None, id = None, parent = None, last_move = None):
        self.state = state
        self.parent = parent
        self.is_fully_expanded = False
        self.children = []
        self.last_move = last_move

        self.children_info = None
        self.visit = 0
        self.value = 0
        self.best_child_visit = 0
        self.id = id

    def best_child(self, exploration_weight):
        best_index = self.children_info.max_puct_score_index(math.sqrt(self.visit), exploration_weight)
        return self.children[best_index]

    def get_state(self, copy_full_stack, claim_draw):
        if copy_full_stack:
            self.state = self.parent.state.perform_move(self.last_move, copy_full_stack=True, claim_draw=claim_draw)
        elif not self.state:
            self.state = self.parent.state.perform_move(self.last_move, claim_draw=claim_draw)

    def expand_specific(self, move, id):
        child_node = MonteCarloNode(parent=self, id=id, last_move=move)
        self.children.append(child_node)

        return child_node

    def expand(self, policies, init_value):
        legal_moves = self.state.get_legal_moves()
        for id, move in enumerate(legal_moves):
            self.expand_specific(move, id)

        self.children_info = NodeInformation(policies, init_value)
        self.is_fully_expanded = True

    def backpropagate(self, reward):
        self.visit += 1
        self.value += reward
        if self.parent:
            self.parent.children_info.add(self.id, reward)
            self.parent.backpropagate(-reward)

            self.parent.best_child_visit = max(self.parent.best_child_visit, self.visit)

            # check sticky result
            if self.parent.state.has_sticky_result:
                return
            if not self.state.can_have_sticky_result:
                self.parent.state.can_have_sticky_result = False
                return
            if self.state.has_sticky_result:
                if self.state.result == -1:
                    self.parent.state.result = 1  # win
                    self.parent.state.has_sticky_result = True
                    return

                if self.parent.state.result is None:
                    self.parent.state.result = -self.state.result
                elif self.parent.state.result == -self.state.result:
                    self.parent.children_info.add_sticky_result(self.id)
                    if self.parent.children_info.has_sticky_result():
                        self.parent.state.has_sticky_result = True
                else: # self.parent.state.result != -self.state.result:
                    self.parent.state.can_have_sticky_result = False
