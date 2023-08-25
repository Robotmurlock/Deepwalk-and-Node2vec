import os

# Project conventions
CHECKPOINT_DIRNAME = 'checkpoints'
TB_LOGS_DIRNAME = 'tb_logs'
RUN_HISTORY = 'run_history'

DATE_FORMAT = '%Y-%m-%d'
TIME_FORMAT = '%H-%M-%S.%f'
DATETIME_FORMAT = f'{DATE_FORMAT}_{TIME_FORMAT}'


def get_tb_logs_dirpath(output_dir: str) -> str:
    return os.path.join(output_dir, TB_LOGS_DIRNAME)


def get_tb_logs_experiment_path(output_dir: str, experiment: str) -> str:
    return os.path.join(get_tb_logs_dirpath(output_dir), experiment)


def get_checkpoints_dirpath(output_dir: str) -> str:
    return os.path.join(output_dir, CHECKPOINT_DIRNAME)


def get_checkpoints_experiment_path(output_dir: str, experiment: str) -> str:
    return os.path.join(get_checkpoints_dirpath(output_dir), experiment)


def get_run_history_dirpath(output_dir: str) -> str:
    return os.path.join(output_dir, RUN_HISTORY)


def get_run_history_experiment_path(output_dir: str, experiment: str) -> str:
    return os.path.join(get_run_history_dirpath(output_dir), experiment)
