import random
from abc import ABC, abstractmethod
from typing import List, Optional, Dict

import networkx as nx

from shallow_encoders.word2vec.dataloader.registry import register_dataset


class RandomWalk(ABC):
    def __init__(self, graph: nx.Graph, length: int):
        assert length >= 1, 'Minimum walk length is 1!'

        self._graph = graph
        self._length = length

    @abstractmethod
    def walk(self, node: str) -> str:
        pass


class DeepWalk(RandomWalk):
    def walk(self, node: str) -> str:
        walk_nodes: List[str] = [node]

        while len(walk_nodes) < self._length:
            neighbours = list(self._graph.neighbors(node))
            child = random.choice(neighbours)
            walk_nodes.append(child)

            node = child

        return ' '.join(walk_nodes)


def random_walk_factory(name: str, graph: nx.Graph, length: int, additional_params: Optional[dict] = None) -> RandomWalk:
    name = name.lower()
    if additional_params is None:
        additional_params = {}

    SUPPORTED_METHODS = {
        'deepwalk': DeepWalk
    }

    assert name in SUPPORTED_METHODS, f'Unknown method "{name}". Supported: {list(SUPPORTED_METHODS.keys())}'


    return SUPPORTED_METHODS[name](
        graph=graph,
        length=length,
        **additional_params
    )


class RandomWalkDataset:
    def __init__(self, graph: nx.Graph, walks_per_node: int, walk_length: int, method: str = 'deepwalk'):
        self._graph = graph
        self._nodes = list(self._graph)
        self._walk_generator = random_walk_factory(name=method, graph=graph, length=walk_length)
        self._walks_per_node = walks_per_node

        # State
        self._index = 0

    @property
    def graph(self) -> nx.Graph:
        return self._graph

    def _get_current_node(self) -> str:
        return self._nodes[self._index // self._walks_per_node]

    def __len__(self) -> int:
        return len(self._graph) * self._walks_per_node

    def __iter__(self) -> 'RandomWalkDataset':
        self._index = 0
        return self

    def __next__(self):
        if self._index >= len(self):
            raise StopIteration('Finished.')

        node = self._get_current_node()
        walk = self._walk_generator.walk(node)
        self._index += 1
        return walk


@register_dataset('graph_triplets')
class GraphTriplets(RandomWalkDataset):
    NUM_CLUSTERS = 2

    def __init__(self, walks_per_node: int, walk_length: int, method: str = 'deepwalk'):
        graph = nx.Graph()

        labels = {}
        for i in range(self.NUM_CLUSTERS):
            prefix = chr(ord('a') + i)
            graph.add_edge(f'{prefix}1', f'{prefix}2')
            graph.add_edge(f'{prefix}2', f'{prefix}3')
            for suffix in ['1', '2', '3']:
                labels[f'{prefix}{suffix}'] = str(i)

        super().__init__(
            graph=graph,
            walks_per_node=walks_per_node,
            walk_length=walk_length,
            method=method
        )

        self._labels = labels

    @property
    def labels(self) -> Dict[str, str]:
        return self._labels


@register_dataset('graph_karate_club')
class KarateClubDataset(RandomWalkDataset):
    def __init__(self, walks_per_node: int, walk_length: int, method: str = 'deepwalk'):
        graph = nx.karate_club_graph()
        mapping = {node: f'n{node:02d}' for node in graph.nodes}  # Convert integers to strings
        graph = nx.relabel_nodes(graph, mapping)

        super().__init__(
            graph=graph,
            walks_per_node=walks_per_node,
            walk_length=walk_length,
            method=method
        )

        self._labels = {  # Reference: https://github.com/freditation/karate-club/blob/master/karate.labels
            'n00': '1', 'n01': '1', 'n02': '1', 'n03': '1', 'n04': '1',
            'n05': '1', 'n06': '1', 'n07': '1', 'n08': '1', 'n09': '2',
            'n10': '1', 'n11': '1', 'n12': '1', 'n13': '1', 'n14': '2',
            'n15': '2', 'n16': '1', 'n17': '1', 'n18': '2', 'n19': '1',
            'n20': '2', 'n21': '1', 'n22': '2', 'n23': '2', 'n24': '2',
            'n25': '2', 'n26': '2', 'n27': '2', 'n28': '2', 'n29': '2',
            'n30': '2', 'n31': '2', 'n32': '2', 'n33': '2'
        }

    @property
    def labels(self) -> Dict[str, str]:
        return self._labels
