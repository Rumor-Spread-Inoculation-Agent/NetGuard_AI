"""
utils.py

Helper utilities for the project (log saving, node lists, and GNN encoding stub).
"""

import datetime
from typing import List
from environment import RumorEnv
import os
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Save a transition"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Randomly sample a batch of experiences"""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# --- 2. LOGGING UTILITY (Needed for GUI) ---
def save_log_text(text, filename="rumor_log.txt"):
    """Saves the text from the GUI log to a file."""
    try:
        with open(filename, "w", encoding='utf-8') as f:
            f.write(text)
        print(f"Log saved to {filename}")
    except Exception as e:
        print(f"Error saving log: {e}")
    
def current_time_str():
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')


def save_log_text(log_text: str, filename: str = None) -> str:
    """Save log_text to filename (if provided) or to timestamped file. Returns filepath."""
    if filename is None:
        filename = f'simulation_log_{current_time_str()}.txt'
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(log_text)
    return filename


def get_infected_nodes_from_env(env: RumorEnv) -> List[int]:
    return [n for n, s in env.status.items() if s == RumorEnv.INF]


def get_susceptible_nodes_from_env(env: RumorEnv) -> List[int]:
    return [n for n, s in env.status.items() if s == RumorEnv.SUS]


def encode_graph_state_for_gnn(graph, state):
    """
    Stub: encode graph+state into node features for GNN.
    Example per-node features: [is_sus, is_inf, is_ino, degree]
    Returns list of feature lists (sorted by node id).
    """
    node_features = []
    for node in sorted(graph.nodes()):
        s = state[node]
        degree = graph.degree[node]
        features = [
            1 if s == RumorEnv.SUS else 0,
            1 if s == RumorEnv.INF else 0,
            1 if s == RumorEnv.INO else 0,
            degree
        ]
        node_features.append(features)
    return node_features
