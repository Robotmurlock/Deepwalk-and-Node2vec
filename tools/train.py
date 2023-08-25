import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from word2vec.utils.func import pairwise_cosine_similarity

from config_parser import GlobalConfig, print_config_tree
from tools.common import CONFIG_PATH


@hydra.main(config_path=CONFIG_PATH, config_name='w2v_abcde.yaml')
def main(cfg: DictConfig) -> None:
    print_config_tree(cfg)
    cfg: GlobalConfig = OmegaConf.to_object(cfg)
    dataset = cfg.datamodule.instantiate_dataset()
    dataloader = cfg.datamodule.instantiate_dataloader(dataset=dataset)
    # noinspection PyTypeChecker
    pl_trainer = cfg.instantiate_trainer(dataset=dataset)

    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(cfg.path.output_dir, 'tb_logs'),
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
                dirpath=os.path.join(cfg.path.output_dir, 'checkpoints', cfg.train.experiment),
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