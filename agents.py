"""
agents.py

Contains agent implementations / stubs for the Rumor Spread project.

Provides:
 - BaseAgent
 - RandomAgent
 - HeuristicAgent
 - MCTSAgent
 - RLDQLAgent (stub)
 - RLGNNAgent (stub)
 - RLAgent (alias to RLGNNAgent for backward compatibility)
"""

import random
import copy
import math
from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from environment import RumorEnv


# -------------------------
# Base
# -------------------------
class BaseAgent:
    def __init__(self, name: str = "Base"):
        self.name = name

    def getAction(self, state: Dict, budget: int) -> List[int]:
        """Return a list of node IDs to inoculate (length <= budget)."""
        raise NotImplementedError


# -------------------------
# Helper: extract susceptible node IDs robustly
# -------------------------
def _extract_susceptible_nodes(env: RumorEnv, state: Dict) -> List[int]:
    """
    Accept either:
      - state returned by env.get_state() where state['status'] is a numpy array aligned
        with sorted(env.G.nodes()), or
      - a dict mapping node_id -> status (like env.status.copy()).
    Returns a list of node ids currently susceptible.
    """
    if "status" not in state:
        return [n for n, s in env.status.items() if s == RumorEnv.SUS]

    s = state["status"]

    # numpy array case
    if isinstance(s, np.ndarray):
        nodes_sorted = sorted(env.G.nodes())
        idxs = np.where(s == RumorEnv.SUS)[0].tolist()
        return [nodes_sorted[i] for i in idxs]

    # dict case
    if isinstance(s, dict):
        return [n for n, st in s.items() if st == RumorEnv.SUS]

    # fallback: env.status
    return [n for n, st in env.status.items() if st == RumorEnv.SUS]


# -------------------------
# Random agent
# -------------------------
class RandomAgent(BaseAgent):
    def __init__(self, env: Optional[RumorEnv] = None):
        super().__init__("Random")
        self.env = env

    def getAction(self, state: Dict, budget: int) -> List[int]:
        if self.env is None:
            raise RuntimeError("RandomAgent requires env reference (pass env to constructor or assign .env).")
        sus = _extract_susceptible_nodes(self.env, state)
        if not sus:
            return []
        k = min(budget, len(sus))
        # choose actual node IDs
        return list(np.random.choice(sus, size=k, replace=False))


# -------------------------
# Heuristic: highest-degree susceptible nodes
# -------------------------
class HeuristicAgent(BaseAgent):
    def __init__(self, env: RumorEnv):
        super().__init__("Heuristic-Degree")
        self.env = env

    def getAction(self, state: Dict, budget: int) -> List[int]:
        G = self.env.G
        # Prefer authoritative source: env.status
        sus = [n for n, st in self.env.status.items() if st == RumorEnv.SUS]

        # If that happens to be empty (unlikely), try extracting from provided state
        if not sus:
            sus = _extract_susceptible_nodes(self.env, state)

        if not sus:
            return []

        # sort by degree descending and return top-k node ids
        sorted_nodes = sorted(sus, key=lambda n: G.degree[n], reverse=True)
        return sorted_nodes[:budget]


# -------------------------
# MCTS helpers & agent
# -------------------------
class MCTSNode:
    def __init__(self, parent: Optional["MCTSNode"], action: Optional[int] = None):
        self.parent = parent
        self.action = action  # node id that produced this node (None for root)
        self.children: Dict[int, "MCTSNode"] = {}
        self.visit_count: int = 0
        self.total_reward: float = 0.0

    def is_fully_expanded(self, legal_actions: List[int]) -> bool:
        return len(self.children) >= len(legal_actions)


class MCTSAgent(BaseAgent):
    """
    MCTS planning agent (simple rollout-based).
    Returns a list of node-ids to inoculate (planned greedily up to budget).
    """

    def __init__(self, env: RumorEnv, simulations: int = 100, horizon: int = 10):
        super().__init__("MCTS")
        self.env = env
        self.simulations = int(simulations)
        self.horizon = int(horizon)

    def rollout(self, sim_env: RumorEnv) -> float:
        """Random rollout: inoculate random nodes each step, sum new infections (we minimize)."""
        temp = copy.deepcopy(sim_env)
        total_new = 0
        for _ in range(self.horizon):
            sus = [n for n, st in temp.status.items() if st == RumorEnv.SUS]
            if not sus:
                break
            k = min(temp.daily_budget, len(sus))
            chosen = list(np.random.choice(sus, size=k, replace=False))
            temp.inoculate(chosen)
            summary = temp.step()
            total_new += len(summary.get("newly_infected", []))
        return -total_new  # negative because lower new infections is better

    def select_child_by_uct(self, node: MCTSNode, c: float = 1.414) -> Optional[MCTSNode]:
        best = None
        best_score = -float("inf")
        for child in node.children.values():
            if child.visit_count == 0:
                return child
            exploit = child.total_reward / child.visit_count
            explore = c * math.sqrt(math.log(max(1, node.visit_count)) / child.visit_count)
            score = exploit + explore
            if score > best_score:
                best_score = score
                best = child
        return best

    def mcts_search(self, root_env: RumorEnv) -> Optional[int]:
        legal_root = [n for n, st in root_env.status.items() if st == RumorEnv.SUS]
        if not legal_root:
            return None

        root = MCTSNode(parent=None)

        for _ in range(self.simulations):
            sim_env = copy.deepcopy(root_env)
            node = root

            # Selection
            legal_actions = [n for n, st in sim_env.status.items() if st == RumorEnv.SUS]
            while node.children and node.is_fully_expanded(legal_actions):
                node = self.select_child_by_uct(node)
                if node is None:
                    break
                # apply the action that led to child (if any)
                if node.action is not None:
                    sim_env.inoculate([node.action])
                    sim_env.step()
                legal_actions = [n for n, st in sim_env.status.items() if st == RumorEnv.SUS]
                if not legal_actions:
                    break

            # Expansion
            legal_actions = [n for n, st in sim_env.status.items() if st == RumorEnv.SUS]
            unexpanded = [a for a in legal_actions if a not in node.children]
            if unexpanded:
                act = random.choice(unexpanded)
                child = MCTSNode(parent=node, action=act)
                node.children[act] = child
                node = child
                sim_env.inoculate([act])
                sim_env.step()

            # Rollout
            reward = self.rollout(sim_env)

            # Backpropagation
            temp = node
            while temp is not None:
                temp.visit_count += 1
                temp.total_reward += reward
                temp = temp.parent

        if not root.children:
            return random.choice(legal_root)
        best_child = max(root.children.values(), key=lambda c: c.visit_count)
        return best_child.action

    def getAction(self, state: Dict, budget: int) -> List[int]:
        sim_env = copy.deepcopy(self.env)
        chosen: List[int] = []
        for _ in range(budget):
            best = self.mcts_search(sim_env)
            if best is None:
                break
            chosen.append(best)
            sim_env.inoculate([best])
            sim_env.step()
        return chosen


# -------------------------
# RL
# -------------------------
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 128)
        self.outputlayer = nn.Linear(128, output_size)
    def forward(self, x):
        
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))

        return self.outputlayer(x)

class RLDQLAgent(BaseAgent):
    """
    Placeholder stub for an RL agent using Deep Q-Learning.
    Implement training/inference and return top-k node ids by Q-values.
    """
    def __init__(self, env: Optional[RumorEnv] = None):
        super().__init__("RL-DQL")
        self.env = env

    def getAction(self, state: Dict, budget: int) -> List[int]:
        # TODO: implement Q-network forward pass; for now random fallback
        if self.env is None:
            raise RuntimeError("RLDQLAgent requires env reference")
        sus = _extract_susceptible_nodes(self.env, state)
        if not sus:
            return []
        k = min(budget, len(sus))
        return list(np.random.choice(sus, size=k, replace=False))


class RLGNNAgent(BaseAgent):
    """
    Placeholder stub for a GNN-based agent.
    Implement node-embedding + policy/Q head and return top-k node ids.
    """
    def __init__(self, env: Optional[RumorEnv] = None):
        super().__init__("RL-GNN")
        self.env = env

    def getAction(self, state: Dict, budget: int) -> List[int]:
        # TODO: encode graph + state, run GNN, pick top-k nodes by score
        if self.env is None:
            raise RuntimeError("RLGNNAgent requires env reference")
        sus = _extract_susceptible_nodes(self.env, state)
        if not sus:
            return []
        k = min(budget, len(sus))
        return list(np.random.choice(sus, size=k, replace=False))


# Keep backward-compatible name RLAgent -> use GNN stub by default
class RLAgent(RLGNNAgent):
    def __init__(self, env: Optional[RumorEnv] = None):
        super().__init__(env)
