"""
environment.py

Rumor spread environment using a Barabási–Albert graph.

State encoding (compatible with the GUI and agents):
    
    
    .SUS = 0
    RumorEnv.INF = 1
    RumorEnv.INO = 2
"""

import random
from typing import List, Dict, Tuple

import networkx as nx
import numpy as np


class RumorEnv:
    SUS = 0
    INF = 1
    INO = 2

    def __init__(self,
                 n_nodes: int = 120,
                 m_edges: int = 2,
                 p_infect: float = 0.15,
                 seed: int = None,
                 initial_infected: int = 1,
                 daily_budget: int = 5):
        self.n_nodes = int(n_nodes)
        self.m_edges = int(m_edges)
        self.p_infect = float(p_infect)
        self.seed = seed
        self.initial_infected = int(initial_infected)
        self.daily_budget = int(daily_budget)

        self.G: nx.Graph = None
        self.status: Dict[int, int] = {}
        self.day = 0
        self.history = []

        self._build_graph()

    def _build_graph(self):
        # build BA graph
        '''self.G = nx.barabasi_albert_graph(self.n_nodes, self.m_edges, seed=self.seed)
        # statuses: SUS / INF / INO
        self.status = {n: RumorEnv.SUS for n in self.G.nodes()}
        # choose initial infected nodes (ensure we don't ask for more than available)
        k = min(self.initial_infected, len(self.G.nodes()))
        starts = random.sample(list(self.G.nodes()), k=k)
        for s in starts:
            self.status[s] = RumorEnv.INF
        self.day = 0
        self.history = []'''

        # A TOY PROBLEM TO SHOW THE PERFORMANCE UPGRADE OF MCTS OVER HEURISTIC AND RANDOM AGENTS.
        self.G = nx.barbell_graph(5, 1)

        # 2. Set all nodes to Susceptible
        self.status = {n: RumorEnv.SUS for n in self.G.nodes()}

        # 3. Manually infect a "leaf" node on one side (e.g., node 0)
        # This is far from the bridge (node 5)
        self.status[0] = RumorEnv.INF
        # --- END TEST MODIFICATION ---

        self.day = 0
        self.history = []

    def reset(self):
        self._build_graph()
        return self.get_state()

    def inoculate(self, nodes: List[int]) -> int:
        """Mark listed nodes as inoculated (INO). Returns how many actually changed."""
        changed = 0
        for n in nodes:
            if n in self.status and self.status[n] == RumorEnv.SUS:
                self.status[n] = RumorEnv.INO
                changed += 1
        return changed

    def step(self) -> Dict:
        """
        Advance one time-step (day): infected nodes try to infect susceptible neighbors.
        Returns dict with day, counts and newly_infected list.
        """
        newly_infected = []
        for u in list(self.G.nodes()):
            if self.status[u] == RumorEnv.INF:
                for v in self.G.neighbors(u):
                    if self.status[v] == RumorEnv.SUS:
                        if random.random() < self.p_infect:
                            newly_infected.append(v)

        # apply infections
        for v in newly_infected:
            self.status[v] = RumorEnv.INF

        self.day += 1
        counts = self.counts()
        self.history.append((self.day, dict(counts)))
        return {'day': self.day, 'counts': counts, 'newly_infected': newly_infected}

    def counts(self) -> Dict[str, int]:
        sus = sum(1 for s in self.status.values() if s == RumorEnv.SUS)
        inf = sum(1 for s in self.status.values() if s == RumorEnv.INF)
        ino = sum(1 for s in self.status.values() if s == RumorEnv.INO)
        return {'susceptible': sus, 'infected': inf, 'inoculated': ino}

    def get_state(self) -> Dict:
        """
        Return compact state: adjacency matrix, status vector (sorted by node id), day.
        Keep indices consistent: sorted(self.G.nodes()).
        """
        A = nx.to_numpy_array(self.G, dtype=np.float32)
        statuses = np.array([self.status[n] for n in sorted(self.G.nodes())], dtype=np.int8)
        return {'adj': A, 'status': statuses, 'day': self.day}

    def node_list(self) -> List[int]:
        return list(sorted(self.G.nodes()))
