"""
agents.py

Contains agent implementations for the Rumor Spread project.

Provides:
 - BaseAgent
 - RandomAgent
 - HeuristicAgent
 - MCTSAgent
 - RLDQLAgent 
 - GNNAgent 
"""

import random
import copy
import math
from typing import List, Dict, Optional
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple
import numpy as np
from environment import RumorEnv
from torch_geometric.nn import SAGEConv
import torch.optim as optim


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
            raise RuntimeError("RandomAgent requires env reference.")
        
        # Filter explicitly for SUSCEPTIBLE nodes only.
        # This prevents picking nodes that are already inoculated or infected.
        susceptible_nodes = [n for n, status in self.env.status.items() if status == RumorEnv.SUS]
        
        # If no one is left to vaccinate, do nothing
        if not susceptible_nodes:
            return []
            
        # Pick k unique nodes from the VALID pool only
        k = min(budget, len(susceptible_nodes))
        selected = random.sample(susceptible_nodes, k)
        
        return [int(x) for x in selected]


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
        self.layer1 = nn.Linear(input_size, 128) #first layer receiving input and learns the weight vectors and convert the inputs to 128 numbers
        self.layer2 = nn.Linear(128, 128) #second layer that takes the 128 numbers and learns complex patterns and produces 128 numbers again
        self.outputlayer = nn.Linear(128, output_size) #takes those 128 numbers and returns a vector of size output size

    #this function defines the pass input->output
    def forward(self, x):

        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))

        return self.outputlayer(x)
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class RLDQLAgent(BaseAgent):
    """
    RL agent using Deep Q-Learning.
    Implement training/inference and return top-k node ids by Q-values.
    """
    def __init__(self, env: Optional[RumorEnv] = None, learning_rate = 0.001):
        super().__init__("RL-DQL")
        self.env = env
        self.epsilon = 1.0
        self.epsilon_min = 1e-2
        self.epsilon_decay = 0.995
        self.input_size = env.n_nodes
        self.output_size = env.n_nodes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(self.input_size, self.output_size).to(self.device)
        self.target_net = DQN(self.input_size, self.output_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayMemory(10000)  # Capacity N
        self.batch_size = 64  # How many memories to learn from at once
        self.gamma = 0.99  # Discount factor (Future vs Now)
        self.target_update = 10  # Update target net every 10 episodes
        import os
        model_path = 'models/dqn_baseline.pth'

        if os.path.exists(model_path):
            # Load the weights from the file
            self.policy_net.load_state_dict(torch.load(model_path, map_location=self.device))

            # Set to Evaluation Mode (tells PyTorch we are not training)
            self.policy_net.eval()

            self.epsilon = 0.0

            print(f"Loaded trained model from {model_path}")
        else:
            print(f"No trained model found at {model_path}. Running with random weights.")


    def getAction(self, state: Dict, budget: int) -> List[int]:
        if self.env is None:
            raise RuntimeError("RLDQLAgent requires env reference")
        # Identify valid (susceptible) nodes
        # We can only inoculate nodes that are currently SUSCEPTIBLE (0)
        # state['status'] is a numpy array e.g., [0, 1, 0, 2...]
        susceptible_mask = (state['status'] == RumorEnv.SUS)
        valid_indices = np.where(susceptible_mask)[0]

        # If no nodes are left to save, return empty
        if len(valid_indices) == 0:
            return []

        # Ensure budget doesn't exceed available nodes
        k = min(budget, len(valid_indices))

        # --- EXPLORATION (Random) ---
        if random.random() < self.epsilon:
            return list(np.random.choice(valid_indices, size=k, replace=False))

        # --- EXPLOITATION (Neural Network) ---
        else:
            # Prepare Input
            # Convert status array to a Float Tensor on the correct device
            # We add a batch dimension [1, 120] because the NN expects batches
            status_tensor = torch.FloatTensor(state['status']).unsqueeze(0).to(self.device)

            # Ask the Network
            with torch.no_grad():  # We are not training, so turn off gradients
                q_values = self.policy_net(status_tensor)  # Shape: [1, 120]

            # Masking
            # The network might be dumb and try to pick an Infected node.
            # We force the Q-values of invalid nodes to be -Infinity.

            # Create a tensor of -inf
            minus_inf = torch.tensor(float('-inf')).to(self.device)

            # We need a boolean mask for INVALID nodes (opposite of susceptible)
            # 1 means valid, 0 means invalid.
            # We want to keep valid Q-values, and overwrite invalid ones.
            valid_mask_tensor = torch.BoolTensor(susceptible_mask).to(self.device)

            # Where valid_mask is True, keep q_values. Where False, set to -inf.
            masked_q_values = torch.where(valid_mask_tensor, q_values, minus_inf)

            # Pick Top-K
            # shape[1] because shape is [1, 120]
            top_k_values, top_k_indices = torch.topk(masked_q_values, k=k)

            # Convert tensors back to a standard Python list of ints
            return top_k_indices[0].cpu().numpy().tolist()

    def train_step(self):
        """
        The Heart of Deep Q-Learning.
        1. Sample a batch of memories.
        2. Calculate the 'Real' Q-value (from Policy Net).
        3. Calculate the 'Target' Q-value (from Target Net + Reward).
        4. Calculate Loss (difference) and update weights.
        """
        # Don't train if we don't have enough memories yet
        if len(self.memory) < self.batch_size:
            return

        # Sample a random batch
        transitions = self.memory.sample(self.batch_size)
        # Unzip the batch (turn a list of Transitions into a Transition of lists)
        batch = Transition(*zip(*transitions))

        # Convert to Tensors (and move to GPU if available)
        # Note: We stack them to create a batch dimension
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)  # Shape [64, 1]
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.FloatTensor(batch.done).to(self.device)

        # Compute Q(s, a) - The Agent's Prediction
        # We pass the state batch through the policy net.
        # .gather(1, action_batch) picks ONLY the Q-value for the action we actually took.
        state_action_values = self.policy_net(state_batch).gather(1, action_batch).squeeze(1)

        # Compute V(s') - The Target Value
        # We use the Target Network to predict the best future value.
        with torch.no_grad():
            # max(1)[0] gives the max Q-value for the next state
            next_state_values = self.target_net(next_state_batch).max(1)[0]

        # Compute Expected Q = Reward + Gamma * V(s')
        # If done is 1, (1-done) becomes 0, so future value is ignored (correct for game over).
        expected_state_action_values = reward_batch + (self.gamma * next_state_values * (1 - done_batch))

        # Compute Loss
        # We use SmoothL1Loss (Huber Loss) which is stable for RL
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)

        # Optimize the Model
        self.optimizer.zero_grad()  # Clear old gradients
        loss.backward()  # Calculate new gradients
        # Clip gradients to prevent exploding gradients (stability trick)
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()  # Update weights

    def update_epsilon(self):
        """Decay exploration rate"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        """Copy weights from Policy Net to Target Net"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

# --- GNN AGENT (GraphSAGE) ---

class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels):
        super(GraphSAGE, self).__init__()
        # Use hidden_dim passed from the agent
        self.conv1 = SAGEConv(in_channels, hidden_dim, aggr='max')
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, aggr='max')
        self.lin = nn.Linear(hidden_dim, out_channels)
        
    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        out = self.lin(h)
        return out.squeeze(1)

class GNNAgent:
    def __init__(self, env, hidden_dim=64, learning_rate=0.002):
        self.env = env
        # Input Features = 3 (Status, Degree, Clustering)
        self.input_feat = 3 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = GraphSAGE(self.input_feat, hidden_dim, 1).to(self.device)
        self.target_net = GraphSAGE(self.input_feat, hidden_dim, 1).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Load model if exists
        import os
        model_path = 'models/gnn_policy.pth'
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                if state_dict['conv1.lin_l.weight'].shape[1] == self.input_feat:
                    self.policy_net.load_state_dict(state_dict)
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                    self.epsilon = 0.0
                    print(f"GNN Agent loaded from {model_path}")
                else:
                    print("Model dimension mismatch (New Features). Retraining from scratch.")
                    self.epsilon = 1.0
            except Exception as e:
                print(f"Could not load GNN model: {e}")
                self.epsilon = 1.0
        else:
            print("No trained GNN model found. Agent will be random unless trained.")
            self.epsilon = 1.0

        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.99

    def _get_graph_data(self, state):
        # Status Feature
        status = np.array(state['status'], dtype=np.float32).reshape(-1, 1)
        
        # CACHED Betweenness (Instant access)
        if hasattr(self.env, 'cached_betweenness') and self.env.cached_betweenness is not None:
            betweenness = self.env.cached_betweenness
             
             
            if hasattr(self.env, 'G'):
                degrees = np.array([d for n, d in self.env.G.degree()], dtype=np.float32)
            else:
                degrees = np.sum(state['adj'], axis=1)
        else:
             # Fallback 
             if hasattr(self.env, 'G'):
                 bet_map = nx.betweenness_centrality(self.env.G, normalized=True)
                 betweenness = np.array([bet_map[n] for n in sorted(self.env.G.nodes())], dtype=np.float32)
                 degrees = np.array([d for n, d in self.env.G.degree()], dtype=np.float32)
             else:
                 degrees = np.sum(state['adj'], axis=1)
                 betweenness = np.zeros_like(degrees)
             
        if degrees.max() > 0: degrees /= degrees.max()
        # Betweenness is already normalized in environment.py

        # Stack Features
        x_data = np.column_stack((status, degrees, betweenness))
        x = torch.tensor(x_data, dtype=torch.float).to(self.device)

        # Edges
        if hasattr(self.env, 'G'):
            edges = list(self.env.G.edges())
            source = [s for s, t in edges] + [t for s, t in edges]
            target = [t for s, t in edges] + [s for s, t in edges]
            edge_index = torch.tensor([source, target], dtype=torch.long).to(self.device)
        else:
            rows, cols = np.where(state['adj'] > 0)
            edge_index = torch.tensor([rows, cols], dtype=torch.long).to(self.device)

        return x, edge_index
    
    def getAction(self, state, budget=1):
        susceptible_mask = (state['status'] == 0)
        valid_indices = np.where(susceptible_mask)[0]

        if len(valid_indices) == 0: return []
        k = min(budget, len(valid_indices))
        
        if random.random() < self.epsilon:
            return list(np.random.choice(valid_indices, size=k, replace=False))
            
        x, edge_index = self._get_graph_data(state)
        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(x, edge_index)
            
        minus_inf = torch.tensor(float('-inf')).to(self.device)
        valid_mask_tensor = torch.tensor(susceptible_mask).to(self.device)
        masked_q = torch.where(valid_mask_tensor, q_values.squeeze(), minus_inf)
        
        if k > 0:
            top_k = torch.topk(masked_q, k)
            return top_k.indices.cpu().numpy().tolist()
        return []