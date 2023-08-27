"""
Torch wrapper for implemented dataset support.
"""
import logging
import re
from collections import Counter
from typing import List, Tuple, Optional, Dict, Iterator

import torch
from nltk.stem import WordNetLemmatizer
from torch.utils.data import IterableDataset
from torchtext.vocab import build_vocab_from_iterator

from shallow_encoders.graph.datasets import RandomWalkDataset
from shallow_encoders.word2vec.dataloader.registry import DATASET_REGISTRY
from shallow_encoders.word2vec.dataloader.w2v_datasets import TestDataset  # This import important for the registry

logger = logging.getLogger('W2VDataset')


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

class W2VDataset(IterableDataset):
    """
    Adapter for Pytorch dataloader.

    Note: Loads dataset into a RAM during vocabulary construction.
    """
    def __init__(
        self,
        dataset_name: str,
        context_radius: int = 5,
        min_word_frequency: int = 20,
        lemmatize: bool = False,
        additional_parameters: Optional[dict] = None
    ):
        """
        Args:
            dataset_name: Dataset name
            context_radius: CBOW and SG context radius (number of words before and after)
            min_word_frequency: Minimum number of word occurrences to add it into the vocabulary
            lemmatize: Perform lemmatization on the sentences before tokenization
            additional_parameters: Dataset specific parameters.
        """
        assert dataset_name in DATASET_REGISTRY, \
            f'Dataset "{dataset_name}" is not supported. Supported: {list(DATASET_REGISTRY.keys())}'

        self._context_radius = context_radius
        self._lemmatize = lemmatize
        self._dataset = DATASET_REGISTRY[dataset_name](**additional_parameters)

        tokenslist = list(self.pipeline)  # IMPORTANT: This loads everything in memory
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

        # State
        self._pipeline_state = None

    @property
    def pipeline(self) -> Iterator:
        """
        Get iterator for post-processed sentences.

        Returns:
            Iterator over post-processed sentences.
        """
        return filter(lambda x: x is not None, map(self.sentence_pipeline, self._dataset))

    def sentence_pipeline(self, sentence: str) -> Optional[List[str]]:
        """
        Performs sentence pipeline.
        - Lemmatization (optional) - NLP specific
        - Tokenization (with filtering - NLP specific)
        - Filtering over short sentences - mostly NLP specific

        Args:
            sentence: Raw sentence

        Returns:
            List of postprocessed words
        """
        sentence = lemmatize_sentence(sentence) if self._lemmatize else sentence
        tokens = tokenize(sentence)
        if 2 * self._context_radius + 1 > len(tokens):
            return None
        return tokens

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

    @property
    def has_labels(self) -> bool:
        """
        Check if dataset supports labels.

        Returns:
            True if dataset supports labels else False
        """
        return isinstance(self._dataset, RandomWalkDataset) and self._dataset.has_labels

    @property
    def labels(self) -> Dict[str, str]:
        """
        Gets dataset word (node) labels.

        Returns:
            Dataset labels
        """
        return self._dataset.labels

    def __iter__(self) -> 'W2VDataset':
        self._pipeline_state = self.pipeline
        return self

    def __next__(self) -> torch.Tensor:
        tokens = next(self._pipeline_state)
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
    test_dataset = TestDataset('train')
    print('Test dataset sentenes')
    for sentence in test_dataset:
        print(sentence)

    torch_test_dataset = W2VDataset(dataset_name='test', min_word_frequency=2, context_radius=1)
    print(f'Vocabulary: {torch_test_dataset.vocab.get_stoi()}')
    print('Samples:')
    for inputs in torch_test_dataset:
        print(inputs)


if __name__ == '__main__':
    run_test()
