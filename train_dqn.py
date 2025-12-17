import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Progress bar
import os
import random
from environment import RumorEnv
from agents import RLDQLAgent


def train():
    # --- HYPERPARAMETERS ---
    NUM_EPISODES = 500  # Total training episodes
    MAX_STEPS = 50  # Max days per episode (prevent infinite loops)
    TARGET_UPDATE_FREQ = 10  # Sync target net every 10 episodes

    # Setup Environment and Agent
    env = RumorEnv(n_nodes=120, m_edges=2, p_infect=0.15, daily_budget=5)
    agent = RLDQLAgent(env)

    # Logging for plots
    rewards_history = []
    infected_history = []
    epsilon_history = []

    print(f"🚀 Starting Training on {agent.device}...")

    # --- TRAINING LOOP ---
    # tqdm is used for a progress bar during training
    for episode in tqdm(range(NUM_EPISODES)):

        # randomizing the probability of infection so that the agent performs reasonably when the probability of infection is altered
        # in the simulator
        current_difficulty = random.uniform(0.05, 0.30)
        env.p_infect = current_difficulty

        # Reset Env
        state_dict = env.reset()
        total_reward = 0

        for step in range(MAX_STEPS):
            # Get Action (Epsilon-Greedy)
            # The agent picks 'budget' number of nodes
            action_nodes = agent.getAction(state_dict, env.daily_budget)

            # Take Action
            # Apply the action in the environment
            env.inoculate(action_nodes)
            summary = env.step()

            # Calculate Reward
            # Goal: Minimize infections.
            # Reward = -1 * (New Infections).
            # Small positive reward for stopping the spread completely.
            new_infections = len(summary['newly_infected'])
            reward = -new_infections

            if new_infections == 0 and summary['counts']['infected'] == 0:
                reward += 10  # Big bonus for clearing the virus

            total_reward += reward

            # Store in Memory
            state_array = state_dict['status']
            next_state_dict = env.get_state()
            next_state_array = next_state_dict['status']

            # We will save the experience for EACH node selected to train efficiently.
            for node_action in action_nodes:
                # Store: (State, Action_Node_ID, Reward, Next_State, Done)
                done = 1 if new_infections == 0 and summary['counts']['infected'] == 0 else 0
                agent.memory.push(state_array, node_action, reward, next_state_array, done)

            # Learn
            agent.train_step()

            # Update state
            state_dict = next_state_dict

            # Check Game Over
            if summary['counts']['infected'] == 0:
                break

        # End of Episode housekeeping
        agent.update_epsilon()

        if episode % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()

        # Logging
        rewards_history.append(total_reward)
        infected_history.append(env.counts()['infected'] + env.counts()['inoculated'])  # Total affected
        epsilon_history.append(agent.epsilon)

    # --- SAVE THE MODEL ---
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(agent.policy_net.state_dict(), 'models/dqn_baseline.pth')
    print("\n💾 Model saved to models/dqn_policy.pth")

    # --- PLOTTING RESULTS ---
    plot_results(rewards_history, epsilon_history)


def plot_results(rewards, epsilons):
    plt.figure(figsize=(12, 5))

    # Total Reward
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title("Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward (Higher is better)")
    plt.grid(True, alpha=0.3) 

    # Epsilon Decay 
    plt.subplot(1, 2, 2) 
    plt.plot(epsilons, color='green')
    plt.title("Epsilon (Exploration)")
    plt.xlabel("Episode")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_results.png')
    print("📊 Graphs saved to training_results.png")
    plt.show()


if __name__ == "__main__":
    train()