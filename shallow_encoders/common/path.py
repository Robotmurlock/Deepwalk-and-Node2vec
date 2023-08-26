"""
Project paths - Paths are absolute and deduced based on the project location on the file system.
"""
import os
from pathlib import Path

# Project paths
ROOT_PATH = str(Path(__file__).parent.parent.parent)
CONFIG_PATH = os.path.join(ROOT_PATH, 'configs')
RUNS_PATH = os.path.join(ROOT_PATH, 'runs')
ASSETS_PATH = os.path.join(ROOT_PATH, 'assets')
