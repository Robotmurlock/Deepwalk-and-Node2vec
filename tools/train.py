"""
Trains Word2Vec type model (SG/CBOW).
"""
import logging
import os.path
import shutil

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from shallow_encoders.common.path import CONFIG_PATH
from tools import conventions
from tools.utils import setup_pipeline

logger = logging.getLogger('Trainer')


def check_train_experiment_history(output_dir: str, dataset_name: str, experiment: str) -> None:
    """
    Checks if trained model already exists. If it already exists then user is asked if he wants to override
    old model (delete old model).

    Args:
        output_dir: Output path
        dataset_name: Dataset name
        experiment: Experiment name
    """
    exp_tb_logs_dirpath = conventions.get_tb_logs_experiment_path(output_dir, dataset_name, experiment)
    exp_checkpoints_dirpath = conventions.get_checkpoints_experiment_path(output_dir, dataset_name, experiment)
    dirpaths = [exp_tb_logs_dirpath, exp_checkpoints_dirpath]

    # Check if there are already some checkpoints or TB logs
    if any(os.path.exists(dirpath) for dirpath in dirpaths):
        logger.warning(f'Experiment "{experiment}" already has some history. Do you want to delete it?')
        response = input(f'Delete "{experiment}" history? [yes/no]   ')
        if response.lower() == 'yes':
            for dirpath in dirpaths:
                if os.path.exists(dirpath):
                    shutil.rmtree(dirpath)



@hydra.main(config_path=CONFIG_PATH, config_name='w2v_sg_abcde.yaml')
def main(cfg: DictConfig) -> None:
    cfg = setup_pipeline(cfg, task='train')
    check_train_experiment_history(
        output_dir=cfg.path.output_dir,
        dataset_name=cfg.datamodule.dataset_name,
        experiment=cfg.train.experiment
    )

    dataset = cfg.datamodule.instantiate_dataset()
    dataloader = cfg.datamodule.instantiate_dataloader(dataset=dataset)
    # noinspection PyTypeChecker
    pl_trainer = cfg.instantiate_trainer(dataset=dataset)

    tb_logger = TensorBoardLogger(
        save_dir=conventions.get_tb_logs_dirpath(cfg.path.output_dir, cfg.datamodule.dataset_name),
        name=cfg.train.experiment
    )

    checkpoints_dirpath = \
        conventions.get_checkpoints_experiment_path(cfg.path.output_dir, cfg.datamodule.dataset_name, cfg.train.experiment)
    trainer = Trainer(
        devices=cfg.train.devices,
        accelerator=cfg.train.accelerator,
        max_epochs=cfg.train.max_epochs,
        logger=tb_logger,
        log_every_n_steps=1,
        callbacks=[
            ModelCheckpoint(
                dirpath=checkpoints_dirpath,
                filename='checkpoint_{epoch:06d}_{step:09d}',  # Example: checkpoint_epoch=000009_step=000034030.ckpt
                save_top_k=-1,  # `-1` == saves every checkpoint
                save_last=True
            )
        ]
    )

    trainer.fit(model=pl_trainer, train_dataloaders=dataloader)



if __name__ == '__main__':
    main()
