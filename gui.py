"""
gui.py

PyQt5 GUI for Rumor Control Simulator.
Control panel matches your original layout; added:
 - Save Log button
 - Clear Log button
 - User-as-Agent section placed below Simulation buttons
 - Larger log area
 - Node numbering toggle
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
from agents import HeuristicAgent, RandomAgent, MCTSAgent, RLAgent
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
                color_map.append("#78d2e9")   # very light blue for susceptible/unvisited
                sizes.append(500)             # bigger size
            elif st == RumorEnv.INF:
                color_map.append("red")
                sizes.append(600)
            else:  # cured/inoculated
                color_map.append("green")
                sizes.append(550)

        # Draw edges slightly darker for contrast
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
        # default env (match initial code defaults)
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

        # Left: canvas
        self.canvas = NetworkCanvas(self, width=6, height=6)
        main_layout.addWidget(self.canvas, 3)

        # Right: controls
        right = QVBoxLayout()
        # parameters header
        right.addWidget(QLabel('Parameters'))

        # Row: Nodes & m edges
        hp = QHBoxLayout()
        hp.addWidget(QLabel('Nodes:'))
        self.spin_nodes = QSpinBox(); self.spin_nodes.setRange(10, 2000); self.spin_nodes.setValue(self.env.n_nodes)
        hp.addWidget(self.spin_nodes)
        hp.addWidget(QLabel('m edges:'))
        self.spin_m = QSpinBox(); self.spin_m.setRange(1, 10); self.spin_m.setValue(self.env.m_edges)
        hp.addWidget(self.spin_m)
        right.addLayout(hp)

        # Row: infect p & daily budget
        hp2 = QHBoxLayout()
        hp2.addWidget(QLabel('Infect p:'))
        self.spin_p = QDoubleSpinBox(); self.spin_p.setRange(0.0, 1.0); self.spin_p.setSingleStep(0.01)
        self.spin_p.setValue(self.env.p_infect)
        hp2.addWidget(self.spin_p)
        hp2.addWidget(QLabel('Daily budget:'))
        self.spin_budget = QSpinBox(); self.spin_budget.setRange(0, 100); self.spin_budget.setValue(self.env.daily_budget)
        hp2.addWidget(self.spin_budget)
        right.addLayout(hp2)

        # Agent selection
        right.addWidget(QLabel('Agent'))
        self.btn_agent_heur = QPushButton('Heuristic (degree)')
        self.btn_agent_random = QPushButton('Random')
        self.btn_agent_mcts = QPushButton('MCTS (stub)')
        self.btn_agent_rl = QPushButton('RL-GNN (stub)')
        self.btn_agent_heur.clicked.connect(lambda: self._set_agent('heur'))
        self.btn_agent_random.clicked.connect(lambda: self._set_agent('rand'))
        self.btn_agent_mcts.clicked.connect(lambda: self._set_agent('mcts'))
        self.btn_agent_rl.clicked.connect(lambda: self._set_agent('rl'))
        right.addWidget(self.btn_agent_heur)
        right.addWidget(self.btn_agent_random)
        right.addWidget(self.btn_agent_mcts)
        right.addWidget(self.btn_agent_rl)

        # Simulation controls (kept exactly like your original UI)
        right.addWidget(QLabel('Simulation'))
        self.btn_reset = QPushButton('Reset Environment')
        self.btn_step = QPushButton('Step (one day)')
        self.btn_run = QPushButton('Run (auto)')
        self.btn_stop = QPushButton('Stop')
        self.btn_run_agent_step = QPushButton('Agent: inoculate & step')

        self.btn_reset.clicked.connect(self._reset_env)
        self.btn_step.clicked.connect(self._manual_step)
        self.btn_run.clicked.connect(self._start_auto)
        self.btn_stop.clicked.connect(self._stop_auto)
        self.btn_run_agent_step.clicked.connect(self._agent_action_and_step)

        right.addWidget(self.btn_reset)
        right.addWidget(self.btn_step)
        right.addWidget(self.btn_run)
        right.addWidget(self.btn_stop)
        right.addWidget(self.btn_run_agent_step)

        # User plays as agent (placed below Simulation buttons per your choice)
        right.addWidget(QLabel('User (play as Agent) - enter node numbers comma-separated:'))
        user_h = QHBoxLayout()
        self.user_input = QLineEdit()
        user_h.addWidget(self.user_input)
        self.btn_user_play = QPushButton('Submit (User)')
        self.btn_user_play.clicked.connect(self._user_play)
        user_h.addWidget(self.btn_user_play)
        right.addLayout(user_h)

        # Logging area and buttons
        right.addWidget(QLabel('Log'))
        # bigger log area
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(360)   # increased size
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

        # status
        self.status_label = QLabel('Day: 0 | S:0 I:0 R:0')
        self.score_label = QLabel('Score: 0')
        right.addWidget(self.score_label)
        right.addWidget(self.status_label)

        # checkbox for showing node ids (slow)
        self.check_show_ids = QCheckBox('Show node ids (slow)')
        self.check_show_ids.stateChanged.connect(self._refresh)  # redraw when toggled
        right.addWidget(self.check_show_ids)

        # spacer
        right.addStretch()
        main_layout.addLayout(right, 1)

        self.setLayout(main_layout)

    def _check_game_over(self):
        counts = self.env.counts()
        total = counts['susceptible'] + counts['infected'] + counts['inoculated']
        
        # Stop if all nodes are either infected or inoculated (no susceptible left)
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
            self.log.append('Switched to Heuristic agent')
        elif which == 'rand':
            self.agent = RandomAgent()
            self.log.append('Switched to Random agent')
        elif which == 'mcts':
            self.agent = MCTSAgent(self.env)
            self.log.append('Switched to MCTS (stub) agent')
        elif which == 'rl':
            self.agent = RLAgent()
            self.log.append('Switched to RL-GNN (stub) agent')
        else:
            self.log.append('Unknown agent selection')

    def _reset_env(self):
        nodes = int(self.spin_nodes.value())
        m = int(self.spin_m.value())
        p = float(self.spin_p.value())
        budget = int(self.spin_budget.value())
        # Recreate env with new parameters
        self.env = RumorEnv(n_nodes=nodes, m_edges=m, p_infect=p, initial_infected=1, daily_budget=budget)
        # If agent needs env reference, update (Heuristic/MCTS)
        self.score = 0
        self.score_label.setText("Score: 0")

        if isinstance(self.agent, HeuristicAgent) or isinstance(self.agent, MCTSAgent):
            try:
                self.agent.env = self.env
            except Exception:
                pass
        self.log.append('Environment reset')
        self._refresh()

    def _manual_step(self):
        summary = self.env.step()
        self._update_score()
        if self._check_game_over():
            return

        self.log.append(f"Day {summary['day']}: newly infected {len(summary['newly_infected'])} nodes")
        self._refresh()

    def _agent_action_and_step(self):
        budget = int(self.spin_budget.value())
        state = self.env.get_state()
        # agents expect state and budget
        try:
            nodes = self.agent.getAction(state, budget)
        except TypeError:
            # some stubs might expect env object; try fallback
            try:
                nodes = self.agent.getAction(self.env)
            except Exception:
                nodes = []
        # Ensure nodes are valid ints in graph node space
        nodes = self._sanitize_node_list(nodes)
        changed = self.env.inoculate(nodes)
        self.log.append(f'Inoculated {changed} nodes: {nodes}')
        summary = self.env.step()
        self._update_score()
        if self._check_game_over():
            return
        self.log.append(f"Day {summary['day']}: newly infected {len(summary['newly_infected'])} nodes; counts: {summary['counts']}")
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
        """Convert variety of agent outputs to valid node id list within graph nodes."""
        if nodes is None:
            return []
        # If nodes is numpy array convert to list
        if isinstance(nodes, np.ndarray):
            nodes = nodes.tolist()
        # If nodes is a list of floats/strings attempt to cast to int
        out = []
        for n in nodes:
            try:
                ni = int(n)
                if ni in self.env.G.nodes():
                    out.append(ni)
            except Exception:
                continue
        # unique & keep order
        seen = set()
        filtered = []
        for x in out:
            if x not in seen:
                seen.add(x)
                filtered.append(x)
        return filtered

    def _update_score(self):
        """
        Compute score from scratch using:
        score = 10 * cured_count - 5 * infected_count - 2 * days_passed
        This does NOT accumulate; it computes fresh each time from env counts/day.
        """
        counts = self.env.counts()
        cured = counts.get('inoculated', 0)   # cured/inoculated count
        infected = counts.get('infected', 0)
        days = int(self.env.day)

        score = 10 * cured - 5 * infected - 2 * days
        # store & display
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
            if p == '':
                continue
            try:
                n = int(p)
                if n in self.env.G.nodes():
                    nodes.append(n)
                else:
                    self.log.append(f'Node {n} not in graph; ignored.')
            except ValueError:
                self.log.append(f'Invalid node id: {p}; ignored.')
        nodes = nodes[:self.env.daily_budget]  # enforce budget
        changed = self.env.inoculate(nodes)
        self.log.append(f'User inoculated {changed} nodes: {nodes}')
        summary = self.env.step()
        self._update_score()
        if self._check_game_over():
            return

        self.log.append(f"Day {summary['day']}: newly infected {len(summary['newly_infected'])} nodes; counts: {summary['counts']}")
        self._refresh()

    def _clear_log(self):
        self.log.clear()
        self.log.append('Log cleared.')

    def _save_log(self):
        # open file dialog to choose save location
        path, _ = QFileDialog.getSaveFileName(self, 'Save Log', f'rumor_log_{random.randint(0,9999)}.txt', 'Text Files (*.txt);;All Files (*)')
        if path:
            save_log_text(self.log.toPlainText(), filename=path)
            self.log.append(f'Log saved to {path}')

    def _refresh(self):
        show_labels = self.check_show_ids.isChecked()
        self.canvas.draw_graph(self.env.G, self.env.status, show_labels=show_labels)
        c = self.env.counts()
        self.status_label.setText(f"Day: {self.env.day} | S: {c['susceptible']} I: {c['infected']} R: {c['inoculated']}")

