"""
main.py
Entry point for the Rumor Control Simulator.
"""

from PyQt5.QtWidgets import QApplication
import sys
from gui import MainWindow

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
