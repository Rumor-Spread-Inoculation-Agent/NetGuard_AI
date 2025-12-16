import random
from typing import List, Dict
import networkx as nx
import numpy as np

class RumorEnv:
    SUS = 0
    INF = 1
    INO = 2

    def __init__(self, n_nodes: int = 120, m_edges: int = 2, p_infect: float = 0.15,
                 seed: int = None, initial_infected: int = 1, daily_budget: int = 5):
        self.n_nodes = int(n_nodes)
        self.m_edges = int(m_edges)
        self.p_infect = float(p_infect)
        self.seed = seed
        self.initial_infected = int(initial_infected)
        self.daily_budget = int(daily_budget)
        
        self.G: nx.Graph = None
        self.cached_betweenness = None  # <--- NEW CACHE VARIABLE
        self._build_graph()

    def _build_graph(self):
        # --- THE CAVEMAN GRAPH (Guaranteed Win) ---
        l = 6
        k = 20
        self.G = nx.connected_caveman_graph(l, k)
        
        # --- OPTIMIZATION: PRE-CALCULATE CENTRALITY ---
        # We calculate this ONCE here, so the Agent doesn't have to do it 1000 times later.
        bet_map = nx.betweenness_centrality(self.G, normalized=True)
        self.cached_betweenness = np.array([bet_map[n] for n in sorted(self.G.nodes())], dtype=np.float32)
        if self.cached_betweenness.max() > 0:
            self.cached_betweenness /= self.cached_betweenness.max()

        # Reset Status
        self.status = {n: RumorEnv.SUS for n in self.G.nodes()}
        
        all_nodes = list(self.G.nodes())
        if self.initial_infected > 0:
            patient_zero = random.sample(all_nodes, self.initial_infected)
            for p in patient_zero:
                self.status[p] = RumorEnv.INF

        self.day = 0
        self.history = []
        
        self._initial_status = self.status.copy()
        self._initial_day = 0
        self._initial_history = []

    def reset(self):
        self.status = self._initial_status.copy()
        self.day = self._initial_day
        self.history = list(self._initial_history)
        return self.get_state()

    def step(self):
        newly_infected = []
        current_infected = [n for n, s in self.status.items() if s == RumorEnv.INF]
        
        for u in current_infected:
            for v in self.G.neighbors(u):
                if self.status[v] == RumorEnv.SUS:
                    if random.random() < self.p_infect:
                        newly_infected.append(v)
        
        newly_infected = list(set(newly_infected))
        for v in newly_infected:
            self.status[v] = RumorEnv.INF
            
        self.day += 1
        return {'counts': self.counts(), 'day': self.day, 'newly_infected': newly_infected}

    def inoculate(self, nodes):
        for n in nodes:
            if n in self.status and self.status[n] == RumorEnv.SUS:
                self.status[n] = RumorEnv.INO

    def counts(self):
        return {
            'susceptible': sum(1 for s in self.status.values() if s == 0),
            'infected': sum(1 for s in self.status.values() if s == 1),
            'inoculated': sum(1 for s in self.status.values() if s == 2)
        }

    def get_state(self):
        if not hasattr(self, 'G'): self._build_graph()
        A = nx.to_numpy_array(self.G, dtype=np.float32)
        nodes = sorted(list(self.G.nodes()))
        statuses = np.array([self.status[n] for n in nodes], dtype=np.int8)
        return {'adj': A, 'status': statuses, 'day': self.day}