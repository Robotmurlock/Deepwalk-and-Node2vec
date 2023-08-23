"""
Word2Vec simple dataloader implementation.
"""
import re
from typing import List, Callable, Tuple

import torch
from torch.utils.data import Dataset
from torchtext.datasets import WikiText2, WikiText103
from torchtext.vocab import build_vocab_from_iterator


class TestDataset:
    """
    Test dataset used to test dataloader
    """
    def __init__(self, root: str, split: str):
        _, _ = root, split  # Ignored
        self._sentences = ['a, a, c, b, b', 'hello world! hello world!', 'test here, test there, here there', '.']

        # State
        self._index = 0

    def __iter__(self) -> 'TestDataset':
        self._index = 0
        return self

    def __next__(self):
        if self._index >= len(self._sentences):
            raise StopIteration('Finished.')

        sentence = self._sentences[self._index]
        self._index += 1
        return sentence



SUPPORTED_DATASETS = {
    'wiki-text-2': WikiText2,
    'wiki-test-103': WikiText103,
    'test': TestDataset
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


class W2VDataset(Dataset):
    """
    Adapter for Pytorch dataloader.

    Note: Loads dataset into a RAM.
    """
    def __init__(self, dataset_name: str, split: str, context_radius: int = 5, min_word_frequency: int = 20):
        """
        Args:
            dataset_name: Dataset name
            split: split name (train, val, test)
            context_radius: CBOW and SG context radius (number of words before and after)
            min_word_frequency: Minimum number of word occurrences to add it into the vocabulary
        """
        assert dataset_name in SUPPORTED_DATASETS, f'Dataset "{dataset_name}" is not supported. Supported: {SUPPORTED_DATASETS}'

        self._dataset = SUPPORTED_DATASETS[dataset_name](root=f'/tmp/{dataset_name}', split=split)
        tokenslist = [tokenize(sentence) for sentence in self._dataset]
        self._tokenslist = [tl for tl in tokenslist if 2 * context_radius + 1 <= len(tl)]
        self._vocab = build_vocab_from_iterator(
            iterator=tokenslist,
            specials=['<unk>'],
            min_freq=min_word_frequency
        )
        self._vocab.set_default_index(self._vocab['<unk>'])

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


def create_w2v_collate(mode: str, context_radius: int, max_length: int) -> Callable[[List[torch.Tensor]], Tuple[torch.Tensor, torch.Tensor]]:
    assert mode.lower() in ['sg', 'cbow'], 'Invalid collate mode! Choose "sg" or "cbow"!'
    min_text_length = 2 * context_radius + 1

    def w2v_collate(batch_text: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_inputs, batch_targets = [], []
        for text in batch_text:
            text = text[:max_length]  # clip text
            text_length = text.shape[0]
            assert text_length >= min_text_length, f'Text is too short! [{text_length=}] < [{min_text_length=}]'

            for center_i in range(context_radius, text_length - context_radius):
                if mode == 'sg':
                    # Logic test example:
                    # text_length = 8, context_radius = 3
                    # => center_i in range(3, 8-3) = range(3, 5) = [3, 4]
                    # for center_i = 3 => inputs = words[3], targets = words[0:3] | words[4:7]
                    # for center_i = 4 => inputs = words[4], targets = words[1:4] | words[5:8]

                    inputs = text[center_i]
                    targets = torch.cat([text[center_i - context_radius:center_i], text[center_i + 1:center_i + 1 + context_radius]])
                elif mode == 'cbow':
                    # Logic is "inverse" of SG

                    inputs = torch.cat([text[center_i - context_radius:center_i], text[center_i + 1:center_i + 1 + context_radius]])
                    targets = text[center_i]
                else:
                    raise AssertionError('Invalid Program State!')

                batch_inputs.append(inputs)
                batch_targets.append(targets)

        batch_inputs, batch_targets = torch.stack(batch_inputs), torch.stack(batch_targets)
        return batch_inputs, batch_targets

    return w2v_collate



def run_test() -> None:
    test_dataset = W2VDataset(dataset_name='test', split='train', min_word_frequency=2, context_radius=1)
    print(f'Vocabulary: {test_dataset.vocab.get_stoi()}')
    print('Samples:')
    for i in range(len(test_dataset)):
        print(f'{test_dataset.get_raw(i)} -> {test_dataset[i]}')


if __name__ == '__main__':
    run_test()
