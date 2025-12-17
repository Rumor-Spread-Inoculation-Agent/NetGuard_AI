"""
gui.py

PyQt5 GUI for Rumor Control Simulator.
"""

import sys
import random
from typing import List

from PyQt5 import QtCore
from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QLabel, QVBoxLayout,
    QPushButton, QSpinBox, QDoubleSpinBox, QTextEdit, QCheckBox,
    QLineEdit, QFileDialog
)
import networkx as nx
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from environment import RumorEnv
from agents import HeuristicAgent, RLDQLAgent, MCTSAgent, GNNAgent, RandomAgent
from utils import save_log_text

class NetworkCanvas(FigureCanvas):
    def __init__(self, parent=None, width=100, height=130, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)

    def draw_graph(self, G: nx.Graph, status: dict, show_labels: bool = False):
        """Draws the graph with updated color and size scheme."""
        self.ax.clear()
        pos = nx.spring_layout(G, seed=42, k=7 / np.sqrt(len(G)))

        color_map = []
        sizes = []

        for n in sorted(G.nodes()):
            st = status[n]
            if st == RumorEnv.SUS:
                color_map.append("#78d2e9")   # Susceptible (Blue)
                sizes.append(500)
            elif st == RumorEnv.INF:
                color_map.append("red")       # Infected (Red)
                sizes.append(600)
            else:
                color_map.append("green")     # Inoculated (Green)
                sizes.append(550)

        nx.draw_networkx_edges(G, pos, ax=self.ax, edge_color="#454b7e", alpha=0.6)
        nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=sizes, ax=self.ax)

        if show_labels:
            labels = {n: str(n) for n in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, ax=self.ax)

        self.ax.set_axis_off()
        self.draw()


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Rumor Control Simulator')
        # Default environment
        self.env = RumorEnv(n_nodes=120, m_edges=2, p_infect=0.15, initial_infected=1, daily_budget=5)
        self.agent = HeuristicAgent(self.env)
        self.timer = QtCore.QTimer()
        self.timer.setInterval(800)  # ms
        self.timer.timeout.connect(self._auto_step)
        self.score = 0

        self._init_ui()
        self._refresh()

    def _init_ui(self):
        main_layout = QHBoxLayout()

        # Left: Canvas
        self.canvas = NetworkCanvas(self, width=6, height=6)
        main_layout.addWidget(self.canvas, 3)

        # Right: Controls
        right = QVBoxLayout()
        right.addWidget(QLabel('Parameters'))

        # Row 1: Nodes & Edges
        hp = QHBoxLayout()
        hp.addWidget(QLabel('Nodes:'))
        self.spin_nodes = QSpinBox(); self.spin_nodes.setRange(10, 2000); self.spin_nodes.setValue(self.env.n_nodes)
        hp.addWidget(self.spin_nodes)
        hp.addWidget(QLabel('m edges:'))
        self.spin_m = QSpinBox(); self.spin_m.setRange(1, 10); self.spin_m.setValue(self.env.m_edges)
        hp.addWidget(self.spin_m)
        right.addLayout(hp)

        # Row 2: Probability & Budget
        hp2 = QHBoxLayout()
        hp2.addWidget(QLabel('Infect p:'))
        self.spin_p = QDoubleSpinBox(); self.spin_p.setRange(0.0, 1.0); self.spin_p.setSingleStep(0.01)
        self.spin_p.setValue(self.env.p_infect)
        hp2.addWidget(self.spin_p)
        hp2.addWidget(QLabel('Daily budget:'))
        self.spin_budget = QSpinBox(); self.spin_budget.setRange(0, 100); self.spin_budget.setValue(self.env.daily_budget)
        hp2.addWidget(self.spin_budget)
        right.addLayout(hp2)

        # Row 3: Seed
        hp3 = QHBoxLayout()
        hp3.addWidget(QLabel('Simulation Seed:'))
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(-1, 999999)
        self.spin_seed.setValue(-1)
        self.spin_seed.setSpecialValueText("Random")
        hp3.addWidget(self.spin_seed)
        right.addLayout(hp3)

        # --- AGENT SELECTION ---
        right.addWidget(QLabel('Select Agent:'))
        
        self.btn_agent_heur = QPushButton('Heuristic (Degree)')
        self.btn_agent_rand = QPushButton('Random Agent')
        self.btn_agent_mcts = QPushButton('MCTS')
        self.btn_agent_dqn = QPushButton('DQN Agent (Trained)')
        self.btn_agent_rl = QPushButton('RL-GNN')

        self.btn_agent_heur.clicked.connect(lambda: self._set_agent('heur'))
        self.btn_agent_rand.clicked.connect(lambda: self._set_agent('rand'))
        self.btn_agent_mcts.clicked.connect(lambda: self._set_agent('mcts'))
        self.btn_agent_dqn.clicked.connect(lambda: self._set_agent('dqn'))
        self.btn_agent_rl.clicked.connect(lambda: self._set_agent('rl'))

        right.addWidget(self.btn_agent_heur)
        right.addWidget(self.btn_agent_rand)
        right.addWidget(self.btn_agent_mcts)
        right.addWidget(self.btn_agent_rl)
        right.addWidget(self.btn_agent_dqn)

        # --- SIMULATION CONTROL ---
        right.addWidget(QLabel('Simulation Control'))
        self.btn_reset = QPushButton('Reset Environment')
        self.btn_new_env = QPushButton("New Environment")
        self.btn_run = QPushButton('Run (auto)')
        self.btn_stop = QPushButton('Stop')
        self.btn_run_agent_step = QPushButton('Agent: inoculate & step')

        self.btn_reset.clicked.connect(self._reset_env)
        self.btn_new_env.clicked.connect(self.new_environment)
        self.btn_run.clicked.connect(self._start_auto)
        self.btn_stop.clicked.connect(self._stop_auto)
        self.btn_run_agent_step.clicked.connect(self._agent_action_and_step)

        right.addWidget(self.btn_reset)
        right.addWidget(self.btn_new_env)
        right.addWidget(self.btn_run)
        right.addWidget(self.btn_stop)
        right.addWidget(self.btn_run_agent_step)
        
        # User plays as agent
        right.addWidget(QLabel('User (play as Agent) - enter node numbers comma-separated:'))
        user_h = QHBoxLayout()
        self.user_input = QLineEdit()
        user_h.addWidget(self.user_input)
        self.btn_user_play = QPushButton('Submit (User)')
        self.btn_user_play.clicked.connect(self._user_play)
        user_h.addWidget(self.btn_user_play)
        right.addLayout(user_h)

        # Logging area
        right.addWidget(QLabel('Log'))
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(360)
        right.addWidget(self.log)

        # Log buttons
        log_h = QHBoxLayout()
        self.btn_clear_log = QPushButton('Clear Log')
        self.btn_clear_log.clicked.connect(self._clear_log)
        self.btn_save_log = QPushButton('Save Log')
        self.btn_save_log.clicked.connect(self._save_log)
        log_h.addWidget(self.btn_clear_log)
        log_h.addWidget(self.btn_save_log)
        right.addLayout(log_h)

        # Status & Score
        self.status_label = QLabel('Day: 0 | S:0 I:0 R:0')
        self.score_label = QLabel('Score: 0')
        right.addWidget(self.score_label)
        right.addWidget(self.status_label)

        # Checkbox
        self.check_show_ids = QCheckBox('Show node ids (slow)')
        self.check_show_ids.stateChanged.connect(self._refresh)
        right.addWidget(self.check_show_ids)

        right.addStretch()
        main_layout.addLayout(right, 1)

        self.setLayout(main_layout)

    def new_environment(self):
        nodes = int(self.spin_nodes.value())
        m = int(self.spin_m.value())
        p = float(self.spin_p.value())
        budget = int(self.spin_budget.value())

        self.env = RumorEnv(n_nodes=nodes, m_edges=m, p_infect=p, initial_infected=1, daily_budget=budget)

        # Update agent reference
        try:
            if isinstance(self.agent, (HeuristicAgent, MCTSAgent, RandomAgent)):
                self.agent.env = self.env
        except Exception:
            pass

        self.score = 0
        self.score_label.setText("Score: 0")
        self.log.append("Created NEW environment (new graph).")
        self._refresh()

    def _check_game_over(self):
        counts = self.env.counts()
        total = counts['susceptible'] + counts['infected'] + counts['inoculated']
        if counts['susceptible'] == 0 or counts['infected'] + counts['inoculated'] == total:
            self.timer.stop()
            from PyQt5.QtWidgets import QMessageBox
            msg = QMessageBox()
            msg.setWindowTitle("Game Over")
            msg.setText(f"Game Over!\nScore achieved: {self.score}")
            msg.exec_()
            return True
        return False

    def _set_agent(self, which: str):
        if which == 'heur':
            self.agent = HeuristicAgent(self.env)
            self.log.append('Switched to Heuristic Agent (Degree)')
        elif which == 'rand':
            self.agent = RandomAgent(self.env)
            self.log.append('Switched to Random Agent')
        elif which == 'mcts':
            self.agent = MCTSAgent(self.env)
            self.log.append('Switched to MCTS Agent')
        elif which == 'rl':
            self.agent = GNNAgent(self.env)
            self.log.append('Switched to RL-GNN Agent')
        elif which == 'dqn':
            self.agent = RLDQLAgent(self.env)
            self.log.append('Switched to DQN Agent')
        else:
            self.log.append('Unknown agent selection')

    def _reset_env(self):
        nodes = self.spin_nodes.value()
        budget = self.spin_budget.value()
        user_seed = self.spin_seed.value()
        
        if user_seed == -1:
            import random
            actual_seed = random.randint(0, 10000)
            self.env.seed = actual_seed
            self.log.append(f"Random Seed Generated: {actual_seed}")
        else:
            self.env.seed = user_seed
            self.log.append(f"Fixed Seed Applied: {user_seed}")

        self.env.n_nodes = nodes
        self.env.daily_budget = budget
        self.env.reset()
        
        self.canvas.draw_graph(self.env.G, self.env.status)
        self.log.append(f"Environment reset. Nodes: {nodes}, Budget: {budget}")
        self.btn_run.setEnabled(True)

    def _manual_step(self):
        """Keeping logic just in case, but button is removed."""
        summary = self.env.step()
        self._update_score()
        if self._check_game_over():
            return
        self.log.append(f"Day {summary['day']}: newly infected {len(summary['newly_infected'])} nodes")
        self._refresh()

    def _agent_action_and_step(self):
        budget = int(self.spin_budget.value())
        state = self.env.get_state()
        try:
            nodes = self.agent.getAction(state, budget)
        except TypeError:
            try:
                nodes = self.agent.getAction(self.env)
            except Exception:
                nodes = []
        nodes = self._sanitize_node_list(nodes)
        changed = self.env.inoculate(nodes)
        self.log.append(f'Agent inoculated {changed} nodes: {nodes}')
        summary = self.env.step()
        self._update_score()
        if self._check_game_over():
            return
        self.log.append(f"Day {summary['day']}: infected {len(summary['newly_infected'])}")
        self._refresh()

    def _start_auto(self):
        if not self.timer.isActive():
            self.timer.start()
            self.log.append('Auto-run started')

    def _stop_auto(self):
        if self.timer.isActive():
            self.timer.stop()
            self.log.append('Auto-run stopped')

    def _auto_step(self):
        self._agent_action_and_step()
        counts = self.env.counts()
        if counts['infected'] == 0:
            self.log.append('No infected nodes remain. Stopping auto-run.')
            self.timer.stop()

    def _sanitize_node_list(self, nodes) -> List[int]:
        if nodes is None: return []
        if isinstance(nodes, np.ndarray): nodes = nodes.tolist()
        out = []
        for n in nodes:
            try:
                ni = int(n)
                if ni in self.env.G.nodes():
                    out.append(ni)
            except Exception:
                continue
        seen = set()
        filtered = []
        for x in out:
            if x not in seen:
                seen.add(x)
                filtered.append(x)
        return filtered

    def _update_score(self):
        counts = self.env.counts()
        cured = counts.get('inoculated', 0)
        infected = counts.get('infected', 0)
        days = int(self.env.day)
        score = 10 * cured - 5 * infected - 2 * days
        self.score = score
        self.score_label.setText(f"Score: {self.score}")

    def _user_play(self):
        text = self.user_input.text()
        if text.strip() == '':
            self.log.append('No nodes entered by user.')
            return
        parts = [p.strip() for p in text.split(',')]
        nodes = []
        for p in parts:
            if p == '': continue
            try:
                n = int(p)
                if n in self.env.G.nodes():
                    nodes.append(n)
                else:
                    self.log.append(f'Node {n} not in graph; ignored.')
            except ValueError:
                self.log.append(f'Invalid node id: {p}; ignored.')
        nodes = nodes[:self.env.daily_budget]
        changed = self.env.inoculate(nodes)
        self.log.append(f'User inoculated {changed} nodes: {nodes}')
        summary = self.env.step()
        self._update_score()
        if self._check_game_over(): return
        self.log.append(f"Day {summary['day']}: newly infected {len(summary['newly_infected'])}")
        self._refresh()

    def _clear_log(self):
        self.log.clear()
        self.log.append('Log cleared.')

    def _save_log(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Save Log', f'rumor_log_{random.randint(0,9999)}.txt', 'Text Files (*.txt);;All Files (*)')
        if path:
            save_log_text(self.log.toPlainText(), filename=path)
            self.log.append(f'Log saved to {path}')

    def _refresh(self):
        show_labels = self.check_show_ids.isChecked()
        self.canvas.draw_graph(self.env.G, self.env.status, show_labels=show_labels)
        c = self.env.counts()
        self.status_label.setText(f"Day: {self.env.day} | S: {c['susceptible']} I: {c['infected']} R: {c['inoculated']}")