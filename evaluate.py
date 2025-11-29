import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from environment import RumorEnv
from agents import HeuristicAgent, RLDQLAgent
from gnn_agent import GNNAgent  # Import your GNN agent


def run_evaluation(agent_name, n_episodes=50):
    # Create Env
    env = RumorEnv(n_nodes=120, m_edges=2, p_infect=0.15, daily_budget=5)

    # Load the correct agent
    if agent_name == 'Heuristic':
        agent = HeuristicAgent(env)
    elif agent_name == 'GNN':
        agent = GNNAgent(env)
        # Load the trained model
        agent.policy_net.load_state_dict(torch.load('models/gnn_policy.pth'))
        agent.epsilon = 0.0  # Turn off randomness! Critical!

    final_infections = []

    print(f"Testing {agent_name}...")
    for seed in tqdm(range(n_episodes)):
        # SET THE SEED: This ensures both agents play the EXACT same levels
        random.seed(seed)
        np.random.seed(seed)
        env.seed = seed

        state = env.reset()

        for _ in range(50):  # Max steps
            action = agent.getAction(state, env.daily_budget)
            env.inoculate(action)
            summary = env.step()

            if summary['counts']['infected'] == 0:
                break

        # Record how many people eventually got infected
        final_infections.append(env.counts()['infected'] + env.counts()['inoculated'])

    return final_infections


if __name__ == "__main__":
    # 1. Run the Head-to-Head
    heur_scores = run_evaluation('Heuristic')
    gnn_scores = run_evaluation('GNN')

    # 2. Compare Results
    avg_heur = np.mean(heur_scores)
    avg_gnn = np.mean(gnn_scores)

    print(f"\nRESULTS (Lower is Better):")
    print(f"Heuristic Avg Impact: {avg_heur:.2f}")
    print(f"GNN Avg Impact:       {avg_gnn:.2f}")

    # 3. Plot the Distribution (Box Plot)
    plt.figure(figsize=(8, 6))
    plt.boxplot([heur_scores, gnn_scores], labels=['Heuristic', 'GNN'])
    plt.ylabel('Total Infected + Inoculated')
    plt.title('Agent Comparison (50 Fixed Seeds)')
    plt.savefig('final_comparison.png')
    plt.show()