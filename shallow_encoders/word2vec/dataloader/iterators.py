"""
Text iterators.
"""
from typing import List


class InMemoryIterator:
    """
    Stores all sentences in a list and iterates over that list.
    """
    def __init__(self, sentences: List[str]):
        """
        Args:
            sentences: List of sentences
        """
        self._sentences = sentences

        # State
        self._index = 0

    def __iter__(self) -> 'InMemoryIterator':
        self._index = 0
        return self

    def __next__(self):
        if self._index >= len(self._sentences):
            raise StopIteration('Finished.')

        sentence = self._sentences[self._index]
        self._index += 1
        return sentence


class FileIterator:
    """
    Reads sentences from file one by one.
    """
    def __init__(self, path: str):
        """
        Args:
            path: Sentences path
        """
        self._path = path
        self._reader = None

    def __iter__(self) -> 'FileIterator':
        self._reader = open(self._path, 'r', encoding='utf-8')
        return self

    def __next__(self):
        assert self._reader is not None, 'Invalid Program State!'
        sentence = self._reader.readline()
        if not sentence:
            self._reader.close()
            self._reader = None
            raise StopIteration('Finished.')

        return sentence
