Rumor Control Simulator
-----------------------

Files:
- environment.py  : Simulation environment (RumorEnv)
- agents.py       : Agent stubs (Heuristic, Random, MCTS, RL)
- utils.py        : Utilities: log saving, state helpers, GNN encoding stub
- gui.py          : PyQt5 GUI (control panel mirrors original interface)
- main.py         : Entry point to run the GUI

How to run:
1) Install dependencies:
   pip install networkx numpy matplotlib PyQt5

2) Run:
   python main.py

Features:
- Full control panel with Parameters, Agent selection, Simulation controls (Reset, Step, Run, Stop), Agent: inoculate & step button
- User play area placed below Simulation buttons (enter comma-separated node IDs)
- Save and Clear log buttons
- Larger log area
- Node numbering toggle (Show node ids)
- Modular structure so you can implement MCTS and RL agents in agents.py

Notes:
- Agents may expect different method signatures for convenience; the GUI tries both getAction(state, budget) and getAction(env) fallbacks.
- encode_graph_state_for_gnn in utils.py provides a simple node feature stub for GNN use.
