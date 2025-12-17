import numpy as np
import torch
import torch.optim as optim
import random
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

from environment import RumorEnv
from agents import GNNAgent
from utils import ReplayBuffer

# --- CONFIGURATION ---
TRAIN_NODES = 120
TRAIN_BUDGET = 10
NUM_EPISODES = 300      
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05           
EPS_DECAY = 150         
LR = 0.002              
TARGET_UPDATE = 10      
MIN_P = 0.15
MAX_P = 0.30

def train():
    # Setup Environment (Caveman Graph)
    env = RumorEnv(n_nodes=TRAIN_NODES, m_edges=2, p_infect=0.25, daily_budget=TRAIN_BUDGET)
    
    # Setup Agent
    agent = GNNAgent(env, hidden_dim=64, learning_rate=LR)
    
    # Setup Memory
    memory = ReplayBuffer(capacity=10000)
    optimizer = optim.Adam(agent.policy_net.parameters(), lr=LR)
    
    rewards_history = []
    loss_history = []
    
    print(f"Training GNN on COMMUNITY Graph...")
    print(f"Nodes: {TRAIN_NODES} | p_infect: {MIN_P}-{MAX_P}")
    
    for episode in tqdm(range(NUM_EPISODES)):
        # Randomize Difficulty
        env.p_infect = random.uniform(MIN_P, MAX_P)
        
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        # Epsilon Decay
        epsilon = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * episode / EPS_DECAY)
        agent.epsilon = epsilon
        
        while not done and steps < 50:
            # Action
            action = agent.getAction(state, env.daily_budget)
            
            # Step
            env.inoculate(action)
            summary = env.step()
            
            newly_infected = len(summary['newly_infected'])
            current_infected = summary['counts']['infected']
            
            # Reward
            reward = -1.0 * (current_infected + newly_infected)
            if current_infected == 0:
                reward += 100 
                done = True
            
            # Store
            next_state = env.get_state()
            memory.push(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # Training Step (The Heavy Loop)
            if len(memory) >= BATCH_SIZE:
                transitions = memory.sample(BATCH_SIZE)
                optimizer.zero_grad()
                total_batch_loss = 0
                
                # Loop through batch to handle dynamic graphs
                for i in range(BATCH_SIZE):
                    s, a, r, ns, d = transitions[i]
                    
                    agent.policy_net.train()
                    x, edge_index = agent._get_graph_data(s)
                    q_values = agent.policy_net(x, edge_index).squeeze()
                    
                    if len(a) > 0:
                        q_val = q_values[a].mean()
                        
                        with torch.no_grad():
                            nx_x, nx_edge_index = agent._get_graph_data(ns)
                            next_q = agent.target_net(nx_x, nx_edge_index).squeeze().max()
                            target = r + (GAMMA * next_q * (1 - int(d)))
                        
                        # Calculate loss for this sample
                        loss = torch.nn.functional.mse_loss(q_val, torch.tensor(target).to(agent.device).detach())
                        loss.backward()
                        total_batch_loss += loss.item()
                
                optimizer.step()
                loss_history.append(total_batch_loss / BATCH_SIZE)

        # Update Target Network
        if episode % TARGET_UPDATE == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            
        rewards_history.append(total_reward)

    # Save Model
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(agent.policy_net.state_dict(), 'models/gnn_policy.pth')
    print("Model Saved to models/gnn_policy.pth")

    # Plot
    plt.figure(figsize=(10,4))
    plt.plot(rewards_history)
    plt.title("Training Rewards (Community Graph)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig("training_results_community.png")

if __name__ == "__main__":
    train()