"""
agents.py

Contains simple agent stubs:
- BaseAgent
- HeuristicAgent (degree-based)
- RandomAgent
- MCTSAgent (stub)
- RLAgent (stub)
"""

import numpy as np
from typing import List, Dict
from environment import RumorEnv


class BaseAgent:
    def __init__(self, name='Base'):
        self.name = name

    def getAction(self, state: Dict, budget: int) -> List[int]:
        raise NotImplementedError


class HeuristicAgent(BaseAgent):
    """Pick highest-degree susceptible nodes."""
    def __init__(self, env: RumorEnv):
        super().__init__('Heuristic-Degree')
        self.env = env

    def getAction(self, state: Dict, budget: int) -> List[int]:
        G = self.env.G
        cands = [n for n in G.nodes() if self.env.status[n] == RumorEnv.SUS]
        cands_sorted = sorted(cands, key=lambda x: G.degree[x], reverse=True)
        return cands_sorted[:budget]


class RandomAgent(BaseAgent):
    def __init__(self):
        super().__init__('Random')

    def getAction(self, state: Dict, budget: int) -> List[int]:
        sus = np.where(state['status'] == RumorEnv.SUS)[0]
        if len(sus) == 0:
            return []
        choice = list(np.random.choice(sus, size=min(budget, len(sus)), replace=False))
        return choice


class MCTSAgent(BaseAgent):
    """Stub for MCTS/Planning agent. Returns heuristic fallback until implemented."""
    def __init__(self, env: RumorEnv, simulations: int = 100, horizon: int = 3):
        super().__init__('MCTS')
        self.env = env
        self.simulations = simulations
        self.horizon = horizon

    def getAction(self, state: Dict, budget: int) -> List[int]:
        # TODO: implement MCTS (simulate env.step() on copies)
        heuristic = HeuristicAgent(self.env)
        return heuristic.getAction(state, budget)


class RLAgent(BaseAgent):
    """Stub for RL agent using GNN encoding. Returns random for now."""
    def __init__(self):
        super().__init__('RL-GNN')

    def getAction(self, state: Dict, budget: int) -> List[int]:
        sus = np.where(state['status'] == RumorEnv.SUS)[0]
        if len(sus) == 0:
            return []
        return list(np.random.choice(sus, size=min(budget, len(sus)), replace=False))
