from typing import Union, Dict, List

import pytorch_lightning as pl
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from word2vec.loss import NegativeSamplingLoss
from word2vec.model import W2VBase
from word2vec.utils import torch_helper
from word2vec.utils.meter import MetricMeter
from word2vec.utils.sampling import generate_noise_batch


class Word2VecTrainer(pl.LightningModule):
    def __init__(
        self,
        model: W2VBase,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        neg_samples: int,
        vocab_size: int
    ):
        super().__init__()
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._loss_func = NegativeSamplingLoss(proba_input=False)

        self._neg_samples = neg_samples
        self._vocab_size = vocab_size

        self._model = model
        self._meter = MetricMeter()

    @property
    def model(self) -> W2VBase:
        return self._model

    @property
    def optimizer(self) -> Optimizer:
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: Optimizer) -> None:
        self._optimizer = optimizer

    @property
    def scheduler(self) -> Union[LRScheduler, dict]:
        return self._scheduler

    @scheduler.setter
    def scheduler(self, scheduler: Union[LRScheduler, dict]) -> None:
        self._scheduler = scheduler

    def _log_loss(self, loss: Union[torch.Tensor, Dict[str, torch.Tensor]], prefix: str, log_step: bool = True) -> None:
        """
        Helper function to log loss. Options:
        - Single value: logged as "{prefix}/loss"
        - Dictionary: for each key log value as "{prefix}/{key}"

        Args:
            loss: Loss
            prefix: Prefix (train or val)
        """
        assert prefix in ['train', 'val'], f'Invalid prefix value "{prefix}"!'

        if isinstance(loss, dict):
            assert 'loss' in loss, \
                f'When returning loss as dictionary it has to have key "loss". Found: {list(loss.keys())}'
            for name, value in loss.items():
                value = value.detach().cpu()
                assert not torch.isnan(value).any(), f'Got nan value for key "{name}"!'
                self._meter.push(f'{prefix}-epoch/{name}', value)
                if log_step:
                    self.log(f'{prefix}/{name}', value, prog_bar=False)
        else:
            assert not torch.isnan(loss).any(), f'Got nan value!'
            loss = loss.detach().cpu()
            self._meter.push(f'{prefix}-epoch/loss', loss)
            if log_step:
                self.log(f'{prefix}/loss', loss, prog_bar=False)

    def forward(self, inputs: torch.Tensor, outputs: torch.Tensor, proba: bool = True) -> torch.Tensor:
        return self._model(inputs, outputs, proba=proba)

    # noinspection PyUnresolvedReferences
    def training_step(self, batch: List[torch.Tensor], *args, **kwargs) -> torch.Tensor:
        inputs, outputs = batch
        noise = generate_noise_batch(inputs.shape[0], self._neg_samples, self._vocab_size).to(inputs)
        positive_logits = self._model(inputs, outputs, proba=False)
        negative_logits = self._model(inputs, noise, proba=False)

        loss = self._loss_func(positive_logits, negative_logits)

        self._log_loss(loss, prefix='train', log_step=True)
        self.log('epoch/lr', torch_helper.get_optim_lr(self.optimizer))

        # Log metrics
        positive_probas = torch.sigmoid(positive_logits)
        recall = (positive_probas >= 0.5).float().mean()
        self._meter.push('train-metrics/recall', recall)
        negative_probas = torch.sigmoid(negative_logits)
        precision = 1 - (negative_probas >= 0.5).float().mean()
        self._meter.push('train-metrics/precision', precision)

        return loss

    def on_train_epoch_end(self) -> None:
        if self._meter.is_empty:
            return

        # Log loss
        for name, value in self._meter.get_all():
            show_on_prog_bar = name.endswith('/loss')
            self.log(name, value, prog_bar=show_on_prog_bar)

    def configure_optimizers(self):
        self._model.parameters()
        return [self._optimizer], [self._scheduler]
