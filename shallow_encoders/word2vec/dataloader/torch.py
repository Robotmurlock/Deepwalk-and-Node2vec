"""
Torch wrapper for implemented dataset support.
"""
import logging
import re
from collections import Counter
from typing import List, Tuple

import torch
from nltk.stem import WordNetLemmatizer
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator
from tqdm import tqdm

from shallow_encoders.word2vec.dataloader.w2v_datasets import (
    WikiText2Dataset,
    WikiText103Dataset,
    TestDataset,
    ABCDEDataset,
    ShakespeareDataset
)

logger = logging.getLogger('W2VDataset')

SUPPORTED_DATASETS = {
    'wiki-text-2': WikiText2Dataset,
    'wiki-text-103': WikiText103Dataset,
    'test': TestDataset,
    'abcde': ABCDEDataset,
    'shakespeare': ShakespeareDataset
}

def tokenize(text: str) -> List[str]:
    """
    Converts raw sentences into a list of tokens:
    - Converts all upper case letters to lower case;
    - Removes punctuations and non-printable characters;
    - Keeps `<unk>` tokens.

    Note: This is a very naive implementation of tokenizer.

    Args:
        text: Raw text (sentence)

    Returns:
        Tokenized text
    """
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*|<unk>')
    return pattern.findall(text.lower())


def lemmatize_sentence(text: str) -> str:
    """
    Lemmatizes words in text. Examples:
        playing -> play,
        played -> play,
        swimming -> swim,
        stronger -> strong

    :param text: Text.
    :return: Text with lemmatized words.
    """
    text = text.lower()

    lemmatizer = WordNetLemmatizer()
    ws = text.split(' ')
    for tag in ['a', 'r', 'n', 'v']:
        ws = list(map(lambda w: lemmatizer.lemmatize(w, tag), ws))
    return ' '.join(ws)

class W2VDataset(Dataset):
    """
    Adapter for Pytorch dataloader.

    Note: Loads dataset into a RAM.
    """
    def __init__(
        self,
        dataset_name: str,
        split: str,
        context_radius: int = 5,
        min_word_frequency: int = 20,
        lemmatize: bool = False,
        *args, **kwargs
    ):
        """
        Args:
            dataset_name: Dataset name
            split: split name (train, val, test)
            context_radius: CBOW and SG context radius (number of words before and after)
            min_word_frequency: Minimum number of word occurrences to add it into the vocabulary
        """
        assert dataset_name in SUPPORTED_DATASETS, \
            f'Dataset "{dataset_name}" is not supported. Supported: {list(SUPPORTED_DATASETS.keys())}'

        self._dataset = SUPPORTED_DATASETS[dataset_name](split=split, *args, **kwargs)
        sentences = [s for s in self._dataset]  # TODO: Everything is currently loaded in memory for Torch dataset
        logger.info(f'Number of loaded sentences is {len(sentences)}.')

        if lemmatize:
            sentences = [lemmatize_sentence(s) for s in tqdm(sentences, desc='lemmatization', unit='sentence')]

        tokenslist = [tokenize(s) for s in tqdm(sentences, desc='tokenization', unit='sentence')]
        self._tokenslist = [tl for tl in tokenslist if 2 * context_radius + 1 <= len(tl)]
        logger.info(f'Number of loaded sentences is {len(self._tokenslist)}.')
        self._vocab = build_vocab_from_iterator(
            iterator=tokenslist,
            specials=['<unk>'],
            min_freq=min_word_frequency
        )
        logger.info(f'Vocabulary size: {len(self._vocab)}')
        self._vocab.set_default_index(self._vocab['<unk>'])

        # Get words frequency
        self._word_frequency = Counter()
        for tl in tokenslist:
            for word in tl:
                if word not in self._vocab:
                    continue
                self._word_frequency[word] += 1
        self._word_frequency = dict(self._word_frequency)

    def get_n_most_frequent_words(self, n: int) -> Tuple[List[str], List[int]]:
        """
        Get `n` most frequent words from vocabulary.

        Args:
            n: Number of words to fetch

        Returns:
            List of words, list of indices
        """
        wfs = list(self._word_frequency.items())
        wfs = sorted(wfs, key=lambda x: x[1], reverse=True)
        wfs = wfs[:n]
        words = [w for w, _ in wfs]
        indices = [self._vocab[w] for w in words]
        return words, indices

    @property
    def vocab(self):
        """
        Gets vocabulary.

        Returns:
            Vocabulary
        """
        return self._vocab

    def __len__(self) -> int:
        return len(self._tokenslist)

    def get_raw(self, i: int) -> List[str]:
        """
        Gets raw tokenized sentence.

        Args:
            i: Item index

        Returns:
            Tokenized sentence
        """
        return self._tokenslist[i]

    def __getitem__(self, i: int) -> torch.Tensor:
        tokens = self.get_raw(i)
        indices = self._vocab(tokens)
        indices = torch.tensor(indices, dtype=torch.long)
        return indices


class W2VCollateFunctional:
    """
    Performs batch collation. Supports `sg` and `cbow` modes.
    """
    def __init__(self, mode: str, context_radius: int, max_length: int):
        """
        Args:
            mode: Mode sg/cbow
            context_radius: Context radius
            max_length: Maximum length
        """
        assert mode.lower() in ['sg', 'cbow'], 'Invalid collate mode! Choose "sg" or "cbow"!'
        self._mode = mode
        self._context_radius = context_radius
        self._min_text_length = 2 * context_radius + 1
        self._max_length = max_length

    def __call__(self, batch_text: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_inputs, batch_targets = [], []
        for text in batch_text:
            text = text[:self._max_length]  # clip text
            text_length = text.shape[0]
            assert text_length >= self._min_text_length, f'Text is too short! [{text_length=}] < [{self._min_text_length=}]'

            for center_i in range(self._context_radius, text_length - self._context_radius):
                if self._mode == 'sg':
                    # Logic test example:
                    # text_length = 8, context_radius = 3
                    # => center_i in range(3, 8-3) = range(3, 5) = [3, 4]
                    # for center_i = 3 => inputs = words[3], targets = words[0:3] | words[4:7]
                    # for center_i = 4 => inputs = words[4], targets = words[1:4] | words[5:8]

                    inputs = text[center_i:center_i+1]
                    targets = torch.cat([text[center_i - self._context_radius:center_i], text[center_i + 1:center_i + 1 + self._context_radius]])
                elif self._mode == 'cbow':
                    # Logic is "inverse" of SG

                    inputs = torch.cat([text[center_i - self._context_radius:center_i], text[center_i + 1:center_i + 1 + self._context_radius]])
                    targets = text[center_i:center_i+1]
                else:
                    raise AssertionError('Invalid Program State!')

                batch_inputs.append(inputs)
                batch_targets.append(targets)

        batch_inputs, batch_targets = torch.stack(batch_inputs), torch.stack(batch_targets)
        return batch_inputs, batch_targets



def run_test() -> None:
    test_dataset = W2VDataset(dataset_name='abcde', split='train', min_word_frequency=2, context_radius=1)
    print(f'Vocabulary: {test_dataset.vocab.get_stoi()}')
    print('Samples:')
    for i in range(len(test_dataset)):
        print(f'{test_dataset.get_raw(i)} -> {test_dataset[i]}')


if __name__ == '__main__':
    run_test()