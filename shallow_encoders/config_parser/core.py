"""
ConfigParser for
"""
import copy
from dataclasses import field
from typing import Iterator
from typing import Union, Optional

from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pydantic.dataclasses import dataclass
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from shallow_encoders.common.path import RUNS_PATH
from shallow_encoders.word2vec.dataloader.torch_dataset import W2VDataset, W2VCollateFunctional
from shallow_encoders.word2vec.model import W2VBase
from shallow_encoders.word2vec.trainer import Word2VecTrainer


@dataclass
class TrainLossConfig:
    negative_samples: int


@dataclass
class TrainConfig:
    experiment: str
    optimizer: dict
    scheduler: dict
    loss: TrainLossConfig
    max_epochs: int
    accelerator: str
    devices: str

    def instantiate_optimizer(self, params: Iterator[nn.Parameter]) -> Optimizer:
        """
        Creates optimizer object. Configuration example:

        Args:
            params: Model parameters

        Returns:
            Optimizer object
        """
        return instantiate(OmegaConf.create(self.optimizer), params=params)

    def instantiate_scheduler(self, optimizer: Optimizer) -> Union[dict, LRScheduler]:
        """
        Creates scheduler object.

        There are two approaches to configure scheduler.
        Option 1 - Instantiate a scheduler object with default PL configuration. Example:
        ```
        _target_: torch.optim.lr_scheduler.StepLR
        step_size: 5
        gamma: 0.2
        ```
        In this case the default interval for scheduler step is one epoch.
        This is ideal for simple StepLR. In case of CyclicLR, WarmUp or CosineAnnealing
        use the second option.

        Option 2 - Instantiate a scheduler object with custom PL configuration. Example:
        ```
        scheduler:
            _target_: torch.optim.lr_scheduler.StepLR
            step_size: 5
            gamma: 0.2
        interval: step
        frequency: 1
        ```

        Args:
            optimizer: Optimizer object

        Returns:
            Scheduler PL configuration
        """
        if '_target_' in self.scheduler:
            # Option 1
            return instantiate(OmegaConf.create(self.scheduler), optimizer=optimizer)

        # Option 2
        assert 'scheduler' in self.scheduler, 'Missing scheduler object in scheduler configuration.'
        scheduler = copy.deepcopy(self.scheduler)
        scheduler['scheduler'] =   instantiate(OmegaConf.create(scheduler['scheduler']), optimizer=optimizer)
        return scheduler


@dataclass
class DatamoduleConfig:
    dataset_name: str
    mode: str
    context_radius: int
    max_length: int
    min_word_frequency: int

    batch_size: int
    num_workers: int

    # Additional dataset config
    lemmatize: bool = False
    additional_parameters: dict = field(default_factory=dict)

    def instantiate_dataset(self) -> W2VDataset:
        """
        Instantiates dateset from config.

        Returns:
            Dataset for training.
        """
        return W2VDataset(
            dataset_name=self.dataset_name,
            context_radius=self.context_radius,
            min_word_frequency=self.min_word_frequency,
            lemmatize=self.lemmatize,
            additional_parameters=self.additional_parameters
        )

    def instantiate_collate_fn(self) -> W2VCollateFunctional:
        """
        Instantiates batch collate function for dataloader.
        This function is mostly implicitly used through `instantiate_dataloader`.

        Returns:
            Collate function.
        """
        return W2VCollateFunctional(
            mode=self.mode,
            context_radius=self.context_radius,
            max_length=self.max_length
        )

    def instantiate_dataloader(
        self,
        dataset: Optional[W2VDataset] = None
    ) -> DataLoader:
        """
        Instantiates training dataloader.

        Args:
            dataset: Dataset

        Returns:
            Dataloader
        """
        dataset = self.instantiate_dataset() if dataset is None else dataset
        collage_fn = self.instantiate_collate_fn()

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collage_fn
        )


@dataclass
class ModelAnalysisConfig:
    checkpoint: str = 'last.ckpt'

    # Closest pairs
    closest_pairs: bool = True
    closest_max_words: int = 100  # In case there are too many words in vocabulary
    closest_pairs_per_word: int = 5

    # Projected embeddings visualization
    visualize_embeddings: bool = True
    visualize_embeddings_max_words: int = 1000
    visualize_skip_unk: bool = True

    # Semantics test
    semantics_test: bool = True


@dataclass
class PathConfig:
    output_dir: str = RUNS_PATH


@dataclass
class GlobalConfig:
    train: TrainConfig
    datamodule: DatamoduleConfig
    model: dict
    analysis: ModelAnalysisConfig = field(default_factory=ModelAnalysisConfig)
    path: PathConfig = field(default_factory=PathConfig)

    def instantiate_model(
        self,
        dataset: Optional[W2VDataset] = None
    ) -> W2VBase:
        """
        Instantiates model (`W2VBase`).

        Args:
            dataset: Dataset (required for vocabulary size)
                - Model is dataset specific

        Returns:
            Model
        """
        dataset = self.datamodule.instantiate_dataset() if dataset is None else dataset
        return instantiate(OmegaConf.create(self.model), vocab_size=len(dataset.vocab))

    def instantiate_trainer(
        self,
        model: Optional[W2VBase] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[LRScheduler] = None,
        dataset: Optional[W2VDataset] = None,
        checkpoint_path: Optional[str] = None
    ) -> Word2VecTrainer:
        """
        Instantiates trainer (PytorchLighting module).

        Args:
            model: Model
            optimizer: Optimizer
            scheduler: Scheduler
            dataset: Dataset
            checkpoint_path: Loads model from checkpoint if defined
                otherwise creates new model

        Returns:
            PL module (trainer)
        """
        dataset = self.datamodule.instantiate_dataset() if dataset is None else dataset
        model = self.instantiate_model(dataset=dataset) if model is None else model
        optimizer = self.train.instantiate_optimizer(model.parameters()) if optimizer is None else optimizer
        scheduler = self.train.instantiate_scheduler(optimizer) if scheduler is None else scheduler
        if checkpoint_path is None:
            return Word2VecTrainer(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                neg_samples=self.train.loss.negative_samples,
                vocab_size=len(dataset.vocab)
            )
        else:

            return Word2VecTrainer.load_from_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                neg_samples=self.train.loss.negative_samples,
                vocab_size=len(dataset.vocab)
            )


# Configuring hydra config store
# If config has `- w2v_config` in defaults then
# full config is recursively instantiated
cs = ConfigStore.instance()
cs.store(name='w2v_config', node=GlobalConfig)
