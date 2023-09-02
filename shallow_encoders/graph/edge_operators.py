"""
Edge embedding operators:
    vector(edge(n1, n2)) = f(vector(n1), vector(n2))
"""
from typing import Callable

import numpy as np


def average(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """
    Calculates average of two node embeddings.

    Args:
        lhs: n1
        rhs: n2

    Returns:
        Average node embedding - edge embedding
    """
    return (lhs + rhs) / 2


def hadamard(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """
    Calculates point-wise multiplication between two nodes.

    Args:
        lhs: n1
        rhs: n2

    Returns:
        point-wise multiplication between two nodes - edge embedding
    """
    return lhs * rhs


def weighted_l1(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """
    Calculates point-wise L1 distance between two nodes.

    Args:
        lhs: n1
        rhs: n2

    Returns:
        point-wise L1 distance between two nodes - edge embedding
    """
    return np.abs(lhs - rhs)


def weighted_l2(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """
    Calculates point-wise L2 distance between two nodes.

    Args:
        lhs: n1
        rhs: n2

    Returns:
        point-wise L2 distance between two nodes - edge embedding
    """
    return (lhs - rhs) ** 2


EdgeOperator = Callable[[np.ndarray, np.ndarray], np.ndarray]


def edge_operator_factory(name: str) -> EdgeOperator:
    """
    Fetched edge operator - fake "factory". Performs validation

    Args:
        name: Operator name

    Returns:
        Edge operator
    """
    name = name.lower()

    EDGE_OPERATORS = {
        'average': average,
        'hadamard': hadamard,
        'weighted_l1': weighted_l1,
        'weighted_l2': weighted_l2
    }

    assert name in EDGE_OPERATORS, f'Operator "{name}" is not supported. Available: {list(EDGE_OPERATORS.keys())}'

    return EDGE_OPERATORS[name]
