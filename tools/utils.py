import os
from datetime import datetime
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from config_parser import GlobalConfig, print_config_tree
from tools.conventions import get_run_history_experiment_path, DATETIME_FORMAT
from common.path import RUNS_PATH


def setup_pipeline(cfg: DictConfig, task: str) -> GlobalConfig:
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
