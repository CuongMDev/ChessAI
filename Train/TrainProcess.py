import random

from Env.GameState import GameState
from MonteCarloTreeSearch.MonteCarloNode import MonteCarloNode
from config.ConfigManager import ConfigManager
from Agent.ExperienceReplay import ExperienceReplay
from MonteCarloTreeSearch.MonteCarloTreeSearch import MonteCarloTreeSearch
from Agent.AgentMemories import AgentMemories
from config.config import TEMPERATURE, GAME_EVALUATE, TEMPERATURE_ENDGAME, TEMPERATURE_DELAY, TEMPERATURE_DECAY, \
    RESIGN_PERCENTAGE, RESIGN_PLAYTHROUGH


def play_with_agent(agent_memory, develop_agent_memory, results, num_game, openings, worker=0):
    # 0: agent win, 1: develop_agent, 2: draw
    config = ConfigManager()
    config.set_mode('eval')

    agent_memories = AgentMemories(agent_memory)
    develop_agent_memories = AgentMemories(develop_agent_memory)
    agents_memories = [agent_memories, develop_agent_memories]

    while True:
        with num_game.get_lock():
            if num_game.value > 0:
                # Split equally between both side
                turn = (num_game.value <= GAME_EVALUATE // 2)
                num_game.value -= 1
                current_game = openings[num_game.value % (GAME_EVALUATE // 2)]
            else:
                break
        game_state = GameState()
        for chess_move in current_game.mainline_moves():
            move = game_state.real_uci_to_move(chess_move.uci())
            game_state = game_state.perform_move(move)

        done = False
        mcts = [None, None]
        for i in range(len(agents_memories)):
            agents_memories[i].change_current_worker_count(1)
            mcts[i] = MonteCarloTreeSearch(agents_memories[i], config, worker, is_training=False, auto_claim_draw=True)
            mcts[i].root = MonteCarloNode(game_state)
            mcts[i].is_start_position = mcts[i].root.state.is_start_position()
            agents_memories[i].change_current_worker_count(-1)

        result = 0
        step_count = 0
        temperature = 0

        while not done:
            agents_memories[turn].change_current_worker_count(1)
            node = mcts[turn].search(temperature)[0]
            agents_memories[turn].change_current_worker_count(-1)

            mcts[turn].update_mcts_root(node)
            step_count += 1

            mcts[not turn].update_mcts_root_from_move(node.last_move)

            result = node.state.result
            done = node.state.is_terminate

            turn = not turn

        winner = not turn
        if result == 0:
            winner = 2 # draw

        with results.get_lock():
            results[winner] += 1 # add winner

def train_with_self(agent_memory, experience_memory, episode, worker=0):
    config = ConfigManager()
    config.set_mode('train')

    agent_memories = AgentMemories(agent_memory)
    experience_replay = ExperienceReplay(experience_memory)

    while True:
        with episode.get_lock():
            if episode.value > 0:
                print(episode.value, end=' ', flush=True)
                episode.value -= 1
            else:
                break

        done = False
        result = 0
        temperature = TEMPERATURE

        step_count = 0
        cache = [] # state, the best policy, reward
        monte_carlo_tree = MonteCarloTreeSearch(agent_memories, config, worker, auto_claim_draw=True)

        while not done:
            best_node, pi = monte_carlo_tree.search(temperature)

            if monte_carlo_tree.root.value / monte_carlo_tree.root.visit * 100 <= -(100 - RESIGN_PERCENTAGE * 2) and random.randint(1, 100) > RESIGN_PLAYTHROUGH: # resign
                result = -1
                break

            cache.append([monte_carlo_tree.root.state.get_train_input()])
            monte_carlo_tree.update_mcts_root(best_node)
            step_count += 1

            result = best_node.state.result
            done = best_node.state.is_terminate

            cache[-1].extend([pi, 1])

            if best_node.state.result_tablebase is not None:
                temperature = 0
            elif step_count >= config.TEMPERATURE_CUTOFF:
                temperature = TEMPERATURE_ENDGAME
            elif step_count >= TEMPERATURE_DELAY:
                temperature = max(0, temperature - TEMPERATURE_DECAY)

        for i in range(len(cache)):
            if (len(cache) - i) % 2:
                cache[i][2] *= -result
            else:
                cache[i][2] *= result
            experience_replay.add_experience(cache[i])

    agent_memories.change_current_worker_count(-1)
