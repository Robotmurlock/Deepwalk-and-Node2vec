"""
Tools utility set.
"""
import os
from datetime import datetime
from pathlib import Path

import matplotlib.colors as mcolors
from omegaconf import DictConfig, OmegaConf

from shallow_encoders.common.path import RUNS_PATH
from shallow_encoders.config_parser import GlobalConfig, print_config_tree
from tools.conventions import get_run_history_experiment_path, DATETIME_FORMAT

MATPLOTLIB_COLORS = list(mcolors.BASE_COLORS) + list(mcolors.CSS4_COLORS)
DEFAULT_WORD_COLOR = 'blue'


def setup_pipeline(cfg: DictConfig, task: str) -> GlobalConfig:
    """
    - Performs fancy rich tree config print.
    - Saves run history in output directory.
    - Instantiates config.

    Args:
        cfg: Config
        task: Task name

    Returns:
        Instantiated config.
    """
    print_config_tree(cfg)

    # Save to run history
    output_dir = cfg.get('output_dir', RUNS_PATH)
    config_dirpath = get_run_history_experiment_path(output_dir, cfg.datamodule.dataset_name, cfg.train.experiment)
    dt = datetime.now().strftime(DATETIME_FORMAT)
    config_path = os.path.join(config_dirpath, f'{task}_{dt}.yaml')
    Path(config_dirpath).mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(OmegaConf.to_yaml(cfg))

    return OmegaConf.to_object(cfg)
