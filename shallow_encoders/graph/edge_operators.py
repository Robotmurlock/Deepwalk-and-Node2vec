from typing import Callable

import numpy as np


def average(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    return (lhs + rhs) / 2


def hadamard(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    return lhs * rhs


def weighted_l1(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    return np.abs(lhs - rhs)


def weighted_l2(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    return (lhs - rhs) ** 2


EdgeOperator = Callable[[np.ndarray, np.ndarray], np.ndarray]


def edge_operator_factory(name: str) -> EdgeOperator:
    name = name.lower()

    EDGE_OPERATORS = {
        'average': average,
        'hadamard': hadamard,
        'weighted_l1': weighted_l1,
        'weighted_l2': weighted_l2
    }

    assert name in EDGE_OPERATORS, f'Operator "{name}" is not supported. Available: {list(EDGE_OPERATORS.keys())}'

    return EDGE_OPERATORS[name]