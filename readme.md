# NetGuard AI: Rumor Containment via Graph Reinforcement Learning 🕸️

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![PyG](https://img.shields.io/badge/PyG-Graph%20Neural%20Networks-green)

**NetGuard AI** is a research-oriented Reinforcement Learning framework designed to optimize the containment of misinformation (rumors) on complex social networks.

By simulating SIR (Susceptible-Infected-Recovered) dynamics on **modular scale-free networks**, this project benchmarks traditional heuristics against AI agents. Our research uncovered the **"Hub Fallacy"**—demonstrating that while degree-based heuristics target influencers, our **GraphSAGE Agent** achieves superior containment by identifying and blocking "bridge" nodes (High Betweenness Centrality) that connect isolated communities.

📄 **[Read the Full Project Report (PDF)](./NetGuardAI.pdf)**
*(Click above to view the detailed methodology, mathematical formulation, and ablation studies)*

---

## 🚀 Key Features

* **Custom Gym Environment:** A stochastic simulation engine for rumor propagation on Caveman and Barabási-Albert graphs.
* **Multi-Agent Support:**
    * **MCTS (Planning):** Monte Carlo Tree Search for look-ahead optimization.
    * **DQN (Model-Free):** Standard Deep Q-Network baseline.
    * **GraphSAGE (SOTA):** Inductive GNN agent with **Max Pooling** aggregation to detect infection fronts.
* **Novel Reward Shaping:** Custom reward functions penalizing both infection rate and total epidemic duration.
* **Interactive Simulator:** A **PyQt5**-based GUI to visualize the spread and agent interventions in real-time.

---

## 🛠️ Tech Stack

* **Core AI:** PyTorch, PyTorch Geometric (PyG)
* **Simulation:** NetworkX
* **Visualization:** Matplotlib, PyQt5
* **Language:** Python 3.10.12

---

## 📥 Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/NetGuard-AI.git](https://github.com/yourusername/NetGuard-AI.git)
cd NetGuard-AI
```

### 2. Set up a Virtual Environment (Recommended)

```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 🖥️ Usage Guide
1. Training the Agents

To train the agents from scratch using the simulation environment:

# Train the GraphSAGE (GNN) Agent:

```bash
python train_gnn.py
```
This will save the model weights to the models/ directory.

# Train the DQN Agent:

```bash
python train_dqn.py
```
This will save the model weights to the models/ directory.

2. Running the Interactive Simulator (GUI)

To watch the agents battle the virus in real-time on a visualized graph:
```bash
python gui.py
```
Controls: Use the menu to select the Agent (GNN, DQN, MCTS, Heuristic) and click "Agent: Innoculate_step."

3. Evaluation & Plotting

To generate the reward curves and win-rate comparison plots:
```bash
python evaluate.py
```
### 📂 Project Structure

NetGuard-AI/
├── agents.py           # Implementation of DQN, GNN, and MCTSAgent classes
├── environment.py      # Custom Gym Environment (SIR dynamics & Graph building)
├── train_gnn.py        # Training loop for GraphSAGE agent
├── train_dqn.py        # Training loop for standard DQN agent
├── gui.py              # PyQt5 application for visualization
├── evaluate.py         # Scripts for plotting results and metrics
├── models/             # Directory where trained model weights (.pth) are saved
├── NetGuardAI.pdf      # Full Project Documentation
├── requirements.txt    # Project dependencies
└── README.md           # Project overview

###📊 Results: The "Hub Fallacy"

Our experiments revealed that Popularity does not equal Virality.

In modular communities, high-degree nodes ("Hubs") are often trapped inside their own clusters. The GraphSAGE Agent outperformed standard heuristics by learning to target Betweenness Centrality—the "Bridges" that allow the virus to jump between clusters.


###👥 Contributors

    [Silajeet Banerjee] - MCTS Agent Design and Implementation, DQN Agent Design and Implementation, GNN-DQN Agent Design and Implementation, GUI, Evaluation.

    [Partner Name] - Heuristic Agent Design and Implementation, Random Agent Design and Implementation, GUI, Environment.
