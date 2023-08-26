"""
Conventions for tools output structure. Structure:

{RUNS_DIRPATH}/
    {dataset_name}/
        {experiment_name}/
            checkpoints/*
            run_history/*
            analysis/*
    tb_logs/
        {dataset_name}/
            {experiment_name}/*

Tensorboard logs are separated on purpose for more friendly experiment names on the board.
To start tensorboard use: `tensorboard --port 7006 --host 0.0.0.0 --logdir {RUNS_DIRPATH}/tb_logs --load_fast=false`
"""
import os

# Project conventions
CHECKPOINT_DIRNAME = 'checkpoints'
TB_LOGS_DIRNAME = 'tb_logs'
RUN_HISTORY_DIRNAME = 'run_history'
ANALYSIS_DIRNAME = 'analysis'

DATE_FORMAT = '%Y-%m-%d'
TIME_FORMAT = '%H-%M-%S.%f'
DATETIME_FORMAT = f'{DATE_FORMAT}_{TIME_FORMAT}'


def get_tb_logs_dirpath(output_dir: str, dataset_name: str) -> str:
    """
    Gets Tensorboard logs directory path.

    Args:
        output_dir: Runs directory
        dataset_name: Dataset name

    Returns:
        Tensorboard directory path
    """
    return os.path.join(output_dir, TB_LOGS_DIRNAME, dataset_name)


def get_tb_logs_experiment_path(output_dir: str, dataset_name: str, experiment: str) -> str:
    """
    Gets tensorboard specific experiment logs directory path

    Args:
        output_dir: Runs directory
        dataset_name: Dataset name
        experiment: experiment name

    Returns:
        Experiment Tensorboard directory path
    """
    return os.path.join(get_tb_logs_dirpath(output_dir, dataset_name), experiment)


def get_experiment_dirpath(output_dir: str, dataset_name: str, experiment: str) -> str:
    """
    Gets path where all experiment files (outputs) are stored.

    Args:
        output_dir: Runs directory
        dataset_name: Dataset name
        experiment: experiment name

    Returns:
        Experiment path
    """
    return os.path.join(output_dir, dataset_name, experiment)


def get_checkpoints_experiment_path(output_dir: str, dataset_name: str, experiment: str) -> str:
    """
    Gets directory path where all experiment are stored.

    Args:
        output_dir: Runs directory
        dataset_name: Dataset name
        experiment: experiment name

    Returns:

    """
    return os.path.join(get_experiment_dirpath(output_dir, dataset_name, experiment), CHECKPOINT_DIRNAME)


def get_checkpoint_path(output_dir: str, dataset_name: str, experiment: str, checkpoint: str) -> str:
    """
    Gets path to the model checkpoint.

    Args:
        output_dir: Runs directory
        dataset_name: Dataset name
        experiment: experiment name
        checkpoint: Checkpoint name

    Returns:
        Path to the experiment checkpoint.
    """
    return os.path.join(get_checkpoints_experiment_path(output_dir, dataset_name, experiment), checkpoint)


def get_run_history_experiment_path(output_dir: str, dataset_name: str, experiment: str) -> str:
    """
    Gets path where all experiment run history (configs) is stored.

    Args:
        output_dir: Runs directory
        dataset_name: Dataset name
        experiment: experiment name

    Returns:
        Path to the experiment run history
    """
    return os.path.join(get_experiment_dirpath(output_dir, dataset_name, experiment), RUN_HISTORY_DIRNAME)


def get_analysis_experiment_path(output_dir: str, dataset_name: str, experiment: str) -> str:
    """
    Gets path where all experiment analysis results are stored.

    Args:
        output_dir: Runs directory
        dataset_name: Dataset name
        experiment: experiment name

    Returns:
        Path to the experiment analysis results
    """
    return os.path.join(get_experiment_dirpath(output_dir, dataset_name, experiment), ANALYSIS_DIRNAME)
