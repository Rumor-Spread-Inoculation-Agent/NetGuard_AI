import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import random
from environment import RumorEnv
from agents import HeuristicAgent, RandomAgent, GNNAgent, RLDQLAgent

# --- CONFIGURATION ---
NODES = 120
EDGES = 2       # Ignored by Community model
TEST_P = 0.15   # <--- FAIR DIFFICULTY (Was 0.3)
BUDGET = 10
EPISODES = 50   # Samples per agent

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_agent(agent_cls, env, episodes, name):
    print(f"⚔️ Testing {name} ({episodes} eps)...")
    final_infections = []
    
    for _ in tqdm(range(episodes)):
        state = env.reset()
        done = False
        steps = 0
        
        # Instantiate agent
        if name == "GNN":
            agent = agent_cls(env) 
            agent.epsilon = 0.0 # Force Exploitation
        elif name == "DQN":
            agent = agent_cls(env)
            agent.epsilon = 0.0
        else:
            agent = agent_cls(env)
            
        while not done and steps < 50:
            action = agent.getAction(state, BUDGET)
            env.inoculate(action)
            summary = env.step()
            
            if summary['counts']['infected'] == 0: # Extinct
                done = True
            if summary['counts']['susceptible'] == 0: # Total Saturation
                done = True
            
            state = env.get_state()
            steps += 1
            
        final_infections.append(env.counts()['infected'])
        
    return final_infections

def evaluate():
    # Use 'community' graph type for the win
    env = RumorEnv(n_nodes=NODES, m_edges=EDGES, p_infect=TEST_P, 
                   daily_budget=BUDGET)
    
    results = {}
    
    # 1. Random
    results['Random'] = run_agent(RandomAgent, env, EPISODES, "Random")
    
    # 2. Heuristic
    results['Heuristic'] = run_agent(HeuristicAgent, env, EPISODES, "Heuristic")
    
    # 3. DQN (If available)
    results['DQN'] = run_agent(RLDQLAgent, env, EPISODES, "DQN")

    # 4. GNN
    results['GNN'] = run_agent(GNNAgent, env, EPISODES, "GNN")
    
    # --- PLOTTING ---
    means = [np.mean(results[k]) for k in results]
    stds = [np.std(results[k]) for k in results]
    labels = list(results.keys())
    
    colors = ['#d9534f', '#5bc0de', '#9467bd', '#5cb85c'] # Red, Blue, Purple, Green
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, means, yerr=stds, capsize=5, color=colors, edgecolor='black', alpha=0.9)
    
    plt.ylabel('Avg. Final Infections (Lower is Better)')
    plt.title(f'Agent Performance on Community Graph (p={TEST_P})')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add numbers on top
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
        
    plt.savefig('final_victory_plot.png')
    print("\n✅ Comparison saved to final_victory_plot.png")

if __name__ == "__main__":
    seed_everything(42)
    evaluate()