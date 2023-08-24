"""
Project paths - Paths are absolute and deduced based on the project location on the file system.
"""
import os
from pathlib import Path

ROOT_PATH = str(Path(__file__).parent.parent)
CONFIG_PATH = os.path.join(ROOT_PATH, 'configs')
