"""
Dynamic dataset registry.
"""
from typing import Type, Callable

DATASET_REGISTRY = {}


def register_dataset(name: str) -> Callable[[Type], Type]:
    """
    Creates a decorator that registers dataset to `DATASET_REGISTRY` dictionary.
    This allows user to dynamically add dataset support.

    Args:
        name: Dataset name

    Returns:
        Class decorator
    """
    assert name not in DATASET_REGISTRY, f'Already registered "{name}"!'

    def decorator(cls: Type) -> Type:
        DATASET_REGISTRY[name] = cls
        return cls

    return decorator
