"""
Implementation of different split algorithms
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np
from sklearn.model_selection import train_test_split


class SplitAlgorithm(ABC):
    """
    Abstract base class for implementing different data splitting algorithms.
    """
    def __init__(self, random_state: Optional[int] = None):
        """
        Args:
            random_state: Seed
        """
        self._random_state = random_state if random_state is not None else 42

    @property
    def random_state(self) -> int:
        return self._random_state

    @random_state.setter
    def random_state(self, random_state: int) -> None:
        self._random_state = random_state

    @abstractmethod
    def split(self, X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Interface definition for split. Takes input data and labels and performs a split algorithm.

        Args:
            X: Input data
            y: labels

        Returns:
            Dictionary of splits.
        """
        pass

    def __call__(self, X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        return self.split(X, y)


class TrainTestRatioSplit(SplitAlgorithm):
    """
    Splits data into train and test based on the `train_ratio`. Supports stratified split.
    """
    def __init__(self, train_ratio: float, stratify: bool = False, test_all: bool = False, random_state: Optional[int] = None):
        """
        Args:
            train_ratio: Train to val ratio
            stratify: Use stratified split
            random_state: Seed
            test_all: Use full dataset for test
        """
        super().__init__(random_state=random_state)
        self._train_ratio = train_ratio
        self._stratify = stratify
        self._test_all = test_all

    def split(self, X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=1 - self._train_ratio,
            stratify=y if self._stratify else None,
            random_state=self._random_state
        )

        return {
            'X_train': X_train.copy(),
            'y_train': y_train.copy(),
            'X_test': X_test.copy() if not self._test_all else X.copy(),
            'y_test': y_test.copy() if not self._test_all else y.copy()
        }


class TrainValTestRatioSplit(SplitAlgorithm):
    """
    Like `TrainTestRatioSplit` but includes also validation split
    """
    def __init__(self, train_ratio: float, val_ratio: float, stratify: bool = False, random_state: Optional[int] = None):
        """
        Args:
            train_ratio: Train to val_test ratio
            val_ratio: Validation to test ratio
            stratify: Use stratified split
            random_state: Seed
        """
        super().__init__(random_state=random_state)
        self._train_ratio = train_ratio
        self._val_ratio = val_ratio
        self._stratify = stratify

    def split(self, X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        X_train, X_val_test, y_train, y_val_test = train_test_split(
            X, y,
            test_size=1 - self._train_ratio,
            stratify=y if self._stratify else None,
            random_state=self._random_state
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_val_test, y_val_test,
            test_size=(1 - self._val_ratio) / (1 - self._train_ratio),
            stratify=y_val_test if self._stratify else None,
            random_state=self._random_state
        )

        return {
            'X_train': X_train.copy(),
            'y_train': y_train.copy(),
            'X_val': X_val.copy(),
            'y_val': y_val.copy(),
            'X_test': X_test.copy(),
            'y_test': y_test.copy(),
        }


class TrainValTestStratifiedNSamplesSplit(SplitAlgorithm):
    """
    Simple algorithm that:
    - Takes `train_samples` per class for train
    - Takes `val_samples` per class for validation
    - Takes `test_samples` per class for tesst
    """
    def __init__(self, train_samples: int, val_samples: int, test_samples: Optional[int] = None, random_state: Optional[int] = None):
        """
        Args:
            train_samples: Number of train samples per class
            val_samples: Number of validation samples per class
            test_samples: Number of test samples per class (optional)
            random_state: Seed
        """
        super().__init__(random_state=random_state)
        self._train_samples = train_samples
        self._val_samples = val_samples
        self._test_samples = test_samples

    def split(self, X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        np.random.seed(self._random_state)

        unique_classes = np.unique(y)
        n_classes = unique_classes.shape[0]
        train_indices = []
        val_indices = []
        test_indices = []

        for label in unique_classes:
            label_indices = np.where(y == label)[0]
            np.random.shuffle(label_indices)

            train_end = self._train_samples
            val_end = train_end + self._val_samples

            train_indices.extend(label_indices[:train_end])
            val_indices.extend(label_indices[train_end:val_end])

            if self._test_samples is not None:
                test_end = val_end + self._test_samples
                test_indices.extend(label_indices[val_end:test_end])
            else:
                test_indices.extend(label_indices[val_end:])

        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        X_test, y_test = X[test_indices], y[test_indices]

        # Validation
        assert X_train.shape[0] == n_classes * self._train_samples, \
            f'{X_train.shape[0]} != {n_classes * self._train_samples}'
        assert y_train.shape[0] == n_classes * self._train_samples, \
            f'{y_train.shape[0]} != {n_classes * self._train_samples}'
        assert X_val.shape[0] == n_classes * self._val_samples, \
            f'{X_val.shape[0]} != {n_classes * self._val_samples}'
        assert y_val.shape[0] == n_classes * self._val_samples, \
            f'{y_val.shape[0]} != {n_classes * self._val_samples}'
        if self._test_samples is not None:
            assert X_test.shape[0] == n_classes * self._test_samples, \
                f'{X_test.shape[0]} != {n_classes * self._test_samples}'
            assert y_test.shape[0] == n_classes * self._test_samples, \
                f'{y_test.shape[0]} != {n_classes * self._test_samples}'

        return {
            'X_train': X_train.copy(),
            'y_train': y_train.copy(),
            'X_val': X_val.copy(),
            'y_val': y_val.copy(),
            'X_test': X_test.copy(),
            'y_test': y_test.copy(),
        }
