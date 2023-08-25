import os.path

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import logging
import shutil

from tools.common import conventions
from tools.common.path import CONFIG_PATH
from tools.utils import setup_pipeline
from word2vec.utils.func import pairwise_cosine_similarity


logger = logging.getLogger('Trainer')


def check_train_experiment_history(output_dir: str, experiment: str) -> None:
    exp_tb_logs_dirpath = conventions.get_tb_logs_experiment_path(output_dir, experiment)
    exp_checkpoints_dirpath = conventions.get_checkpoints_experiment_path(output_dir, experiment)
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
    dataset = cfg.datamodule.instantiate_dataset()
    dataloader = cfg.datamodule.instantiate_dataloader(dataset=dataset)
    # noinspection PyTypeChecker
    pl_trainer = cfg.instantiate_trainer(dataset=dataset)

    check_train_experiment_history(
        output_dir=cfg.path.output_dir,
        experiment=cfg.train.experiment
    )

    tb_logger = TensorBoardLogger(
        save_dir=conventions.get_tb_logs_dirpath(cfg.path.output_dir),
        name=cfg.train.experiment
    )

    trainer = Trainer(
        devices=cfg.train.devices,
        accelerator=cfg.train.accelerator,
        max_epochs=cfg.train.max_epochs,
        logger=tb_logger,
        log_every_n_steps=1,
        callbacks=[
            ModelCheckpoint(
                dirpath=conventions.get_checkpoints_experiment_path(cfg.path.output_dir, cfg.train.experiment),
                filename='checkpoint_{epoch:06d}_{step:09d}',  # Example: checkpoint_epoch=000009_step=000034030.ckpt
                save_top_k=-1  # `-1` == saves every checkpoint
            )
        ]
    )

    trainer.fit(model=pl_trainer, train_dataloaders=dataloader)

    print(dataset.vocab.get_stoi())
    inverse_map = {v: k for k, v in dataset.vocab.get_stoi().items()}
    input_emb = pl_trainer.model.input_embedding
    output_emb = pl_trainer.model.output_embedding
    sim = pairwise_cosine_similarity(input_emb, output_emb)
    print(sim)
    closest = torch.argmax(sim, dim=-1)
    closest = [inverse_map[int(x.item())] for x in closest]
    print(list(zip([inverse_map[i] for i in range(sim.shape[0])], closest)))



if __name__ == '__main__':
    main()