"""
Datasets support.
"""
import os

import pandas as pd

from shallow_encoders.common.path import ASSETS_PATH
from shallow_encoders.word2vec.dataloader.iterators import InMemoryIterator, FileIterator
from shallow_encoders.word2vec.dataloader.registry import register_dataset

@register_dataset('test')
class TestDataset(InMemoryIterator):
    """
    Test dataset used to test dataloader.
    """
    def __init__(self, split: str):
        _ = split
        super().__init__(sentences=[
                'a, a, c, b, b',
                'hello world! hello world!',
                'test here, test there, here there', '.'
            ]
        )


@register_dataset('abcde')
class ABCDEDataset(InMemoryIterator):
    """
    Simple dataset used to test model capability. Model should be able to learn:
    - `a` and `b` go together in a sentence
    - `c` and `d` go together in a sentence
    - `e` goes alone in a sentence
    """
    def __init__(self, split: str):
        _ = split  # Ignored
        super().__init__(
            sentences=[
                'a b a b a b a b a b',  # `a` goes with `b`
                'a b a b a b',
                'b a b a',
                'a b a b a b a b',
                'c d c d c d c d',  # `c` goes with `d`
                'd c d c d c',
                'c d c d c d',
                'e e e e e e e e',  # `e` goes alone
                'e e e'
            ]
        )


class WikiTextDataset(FileIterator):
    """
    Iterator wrapper for wiki datasets. Integrated wiki dataset filename conventions.
    """
    def __init__(self, dataset_name: str, split: str, assets_path: str = ASSETS_PATH):
        """
        Args:
            dataset_name: Dataset name (wiki-something)
            split: Split (usually train)
            assets_path: Path where the dataset is stored (defaults: ASSETS_PATH)
        """
        path = os.path.join(assets_path, dataset_name, f'wiki.{split}.tokens')
        super().__init__(path=path)


@register_dataset('wiki-text-2')
class WikiText2Dataset(WikiTextDataset):
    """
    Concrete Wiki dataset - `wikitext-2`
    """
    def __init__(self, split: str, *args, **kwargs):
        super().__init__(
            dataset_name='wikitext-2',
            split=split,
            *args,
            **kwargs
        )


@register_dataset('wiki-text-103')
class WikiText103Dataset(WikiTextDataset):
    """
    Concrete Wiki dataset - `wikitext-103`
    """
    def __init__(self, split: str, *args, **kwargs):
        super().__init__(
            dataset_name='wikitext-103',
            split=split,
            *args,
            **kwargs
        )


@register_dataset('shakespeare')
class ShakespeareDataset(InMemoryIterator):
    """
    Parses Shakespeare dataset and loads all sentences in memory iterator.
    """
    def __init__(self, split: str, assets_path: str = ASSETS_PATH):
        _ = split
        df = pd.read_csv(os.path.join(assets_path, 'Shakespeare_data.csv'))
        lines = df['PlayerLine'].values.tolist()
        super().__init__(sentences=lines)
