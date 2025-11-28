import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Progress bar
import os

from environment import RumorEnv
from agents import RLDQLAgent


def train():
    # --- HYPERPARAMETERS ---
    NUM_EPISODES = 500  # Total training episodes
    MAX_STEPS = 50  # Max days per episode (prevent infinite loops)
    TARGET_UPDATE_FREQ = 10  # Sync target net every 10 episodes

    # Setup Environment and Agent
    # Note: We use the same parameters as your GUI defaults
    env = RumorEnv(n_nodes=120, m_edges=2, p_infect=0.05, daily_budget=5)
    agent = RLDQLAgent(env)

    # Logging for plots
    rewards_history = []
    infected_history = []
    epsilon_history = []

    print(f"🚀 Starting Training on {agent.device}...")

    # --- TRAINING LOOP ---
    # tqdm creates a nice progress bar
    for episode in tqdm(range(NUM_EPISODES)):

        # 1. Reset Env
        state_dict = env.reset()
        total_reward = 0

        for step in range(MAX_STEPS):
            # 2. Get Action (Epsilon-Greedy)
            # The agent picks 'budget' number of nodes
            action_nodes = agent.getAction(state_dict, env.daily_budget)

            # 3. Take Action
            # We need to map the list of nodes to a single index for the Replay Memory?
            # PROBLEM: Standard DQN selects 1 action. Your getAction selects 5.
            # FIX: For training simple DQN, we will treat the action as "Inoculate Node X".
            # BUT: Your environment requires a list.
            # ADAPTATION: We will store the transition for the FIRST node selected
            # to keep the DQN standard and simple for now.
            # (Multi-action DQN is complex, this is a standard approximation).

            # Apply the action in the environment
            env.inoculate(action_nodes)
            summary = env.step()

            # 4. Calculate Reward
            # Goal: Minimize infections.
            # Reward = -1 * (New Infections).
            # Bonus: Small positive reward for stopping the spread completely.
            new_infections = len(summary['newly_infected'])
            reward = -new_infections

            if new_infections == 0 and summary['counts']['infected'] == 0:
                reward += 10  # Big bonus for clearing the virus

            total_reward += reward

            # 5. Store in Memory
            # We need to reshape state to match what train_step expects
            state_array = state_dict['status']
            next_state_dict = env.get_state()
            next_state_array = next_state_dict['status']

            # DQN usually learns "State -> Single Action".
            # We will save the experience for EACH node selected to train efficiently.
            for node_action in action_nodes:
                # Store: (State, Action_Node_ID, Reward, Next_State, Done)
                done = 1 if new_infections == 0 and summary['counts']['infected'] == 0 else 0
                agent.memory.push(state_array, node_action, reward, next_state_array, done)

            # 6. Learn (Optimization)
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
    torch.save(agent.policy_net.state_dict(), 'models/dqn_policy.pth')
    print("\n💾 Model saved to models/dqn_policy.pth")

    # --- PLOTTING RESULTS ---
    plot_results(rewards_history, infected_history, epsilon_history)


def plot_results(rewards, infected, epsilons):
    plt.figure(figsize=(12, 5))

    # Plot 1: Total Reward over time
    plt.subplot(1, 3, 1)
    plt.plot(rewards)
    plt.title("Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward (Higher is better)")

    # Plot 2: Final Impact (Lower is better)
    plt.subplot(1, 3, 2)
    plt.plot(infected, color='orange')
    plt.title("Total Infected+Inoculated")
    plt.xlabel("Episode")
    plt.ylabel("Count")

    # Plot 3: Epsilon Decay
    plt.subplot(1, 3, 3)
    plt.plot(epsilons, color='green')
    plt.title("Epsilon (Exploration)")
    plt.xlabel("Episode")

    plt.tight_layout()
    plt.savefig('training_results.png')
    print("📊 Graphs saved to training_results.png")
    plt.show()


if __name__ == "__main__":
    train()