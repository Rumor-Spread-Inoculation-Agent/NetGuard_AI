"""
agents.py

Contains simple agent stubs:
- BaseAgent
- HeuristicAgent (degree-based)
- MCTSAgent (stub)
- RLAgent (stub)
- GNNAgent
"""
import random

import numpy as np
import copy
import math
from typing import List, Dict
from environment import RumorEnv


class BaseAgent:
    def __init__(self, name='Base'):
        self.name = name

    def getAction(self, state: Dict, budget: int) -> List[int]:
        raise NotImplementedError


class HeuristicAgent(BaseAgent):
    """Heuristic agent that picks the highest-degree susceptible nodes."""

    def __init__(self, env: RumorEnv):
        super().__init__('Heuristic-Degree')
        self.env = env

    def getAction(self, state: Dict, budget: int) -> List[int]:
        G = self.env.G
        # Find susceptible (uninfected & uncured) nodes
        candidates = [n for n in G.nodes() if self.env.status[n] == RumorEnv.SUS]
        # Sort by node degree (descending)
        sorted_candidates = sorted(candidates, key=lambda x: G.degree[x], reverse=True)
        # Pick top-k nodes as per budget
        return sorted_candidates[:budget]


class RandomAgent(BaseAgent):
    def __init__(self):
        super().__init__('Random')

    def getAction(self, state: Dict, budget: int) -> List[int]:
        sus = np.where(state['status'] == RumorEnv.SUS)[0]
        if len(sus) == 0:
            return []
        choice = list(np.random.choice(sus, size=min(budget, len(sus)), replace=False))
        return choice

class MCTSNode:
    def __init__(self, parent, action: int):
        self.parent = parent
        self.action = action
        self.children = {}
        self.visit_count = 0
        self.total_reward = 0.0
    def fully_expanded(self, legal_moves : List[int]):
        return len(legal_moves) == len(self.children)

class MCTSAgent(BaseAgent):
    """Stub for MCTS/Planning agent. Returns heuristic fallback until implemented."""
    def __init__(self, env: RumorEnv, simulations: int = 100, horizon: int = 15):
        super().__init__('MCTS')
        self.env = env
        self.simulations = simulations
        self.horizon = horizon

    def rollout(self, sim_env):
        temp_env = copy.deepcopy(sim_env)
        total_new_infections = 0
        for _ in range(self.horizon):
            sus_nodes = [n for n, s in temp_env.status.items() if s == RumorEnv.SUS]
            if(not sus_nodes):
                break
            action = list(np.random.choice(sus_nodes, size = min(temp_env.daily_budget, len(sus_nodes)), replace=False))
            temp_env.inoculate(action)
            summary = temp_env.step()
            total_new_infections += len(summary["newly_infected"])
        return -total_new_infections

    def select_best_child_uct(self, node):
        best_score = -float("inf")
        best_child = None
        c = 1.42
        for child in node.children.values():
            if child.visit_count == 0:
                return child
            exploitation_score = child.total_reward / child.visit_count
            exploration_score = c * math.sqrt(math.log(node.visit_count) / child.visit_count)
            uct_score = exploration_score + exploitation_score
            if uct_score > best_score:
                best_child = child
                best_score = uct_score
        return best_child

    def mcts_search(self, root_env):
        legal_moves = [n for n, s in root_env.status.items() if s == RumorEnv.SUS]
        if not legal_moves:
            return None
        root_node = MCTSNode(parent=None, action=None)
        for _ in range(self.simulations):
            sim_env = copy.deepcopy(root_env)
            current_node = root_node
            while current_node.fully_expanded(legal_moves) and current_node.children:
                current_node = self.select_best_child_uct(current_node)
                sim_env.inoculate([current_node.action])

            if not current_node.fully_expanded(legal_moves):
                unexpanded_actions = [m for m in legal_moves if m not in current_node.children]
                action = random.choice(unexpanded_actions)

                new_child = MCTSNode(parent=current_node, action=action)
                current_node.children[action] = new_child
                current_node = new_child

                sim_env.inoculate([action])

            reward = self.rollout(sim_env)
            while current_node is not None:
                current_node.visit_count += 1
                current_node.total_reward += reward
                current_node = current_node.parent

        if not root_node.children:
            return random.choice(legal_moves)  # Failsafe

        best_child = max(root_node.children.values(), key=lambda n: n.visit_count)
        return best_child.action

    def getAction(self, state: Dict, budget: int) -> List[int]:
        sim_env = copy.deepcopy(self.env)
        chosen_nodes = []
        for _ in range(budget):
            best_node = self.mcts_search(sim_env)
            if best_node is None:
                break
            chosen_nodes.append(best_node)

            sim_env.inoculate([best_node])

        return chosen_nodes

class RLAgent(BaseAgent):
    """Stub for RL agent using GNN encoding. Returns random for now."""
    def __init__(self):
        super().__init__('RL-GNN')

    def getAction(self, state: Dict, budget: int) -> List[int]:
        sus = np.where(state['status'] == RumorEnv.SUS)[0]
        if len(sus) == 0:
            return []
        return list(np.random.choice(sus, size=min(budget, len(sus)), replace=False))

