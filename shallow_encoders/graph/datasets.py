"""
Implementation of graph datasets specialized for W2V
"""
import random
from typing import Optional, Dict

import networkx as nx

from shallow_encoders.word2vec.dataloader.registry import register_dataset
from shallow_encoders.graph.random_walk_generator import random_walk_factory


class RandomWalkDataset:
    """
    Implementation of RandomWalk dataset interface.
    Dataset is composed of graph (structure) and graph random walk generator.
    """
    def __init__(
        self,
        graph: nx.Graph,
        walks_per_node: int,
        walk_length: int,
        method: str = 'deepwalk',
        method_params: Optional[dict] = None,
        labels: Optional[Dict[str, str]] = None
    ):
        """
        Args:
            graph: Graph
            walks_per_node: Number of walks to generate for each node
            walk_length: Random walk length
            method: Random walk method
            labels: Graph node labels (optional)
            method_params: Random walk generator specific parameters
        """
        self._graph = graph
        self._nodes = list(self._graph)
        self._labels = labels
        random.shuffle(self._nodes)

        method_params = {} if method_params is None else method_params
        self._walk_generator = random_walk_factory(
            name=method,
            graph=graph,
            length=walk_length,
            additional_params=method_params
        )
        self._walks_per_node = walks_per_node

        # State
        self._index = 0

    @property
    def graph(self) -> nx.Graph:
        """
        Graph getter.

        Returns:
            Graph (networkx)
        """
        return self._graph

    def _get_current_node(self) -> str:
        """
        Gets node that is currently used for random walk generation.

        Returns:
            Current random walk generation node.
        """
        return self._nodes[self._index // self._walks_per_node]

    def __len__(self) -> int:
        return len(self._graph) * self._walks_per_node

    def __iter__(self) -> 'RandomWalkDataset':
        self._index = 0
        return self

    def __next__(self):
        if self._index >= len(self):
            random.shuffle(self._nodes)
            raise StopIteration('Finished.')

        node = self._get_current_node()
        walk = self._walk_generator.walk(node)
        self._index += 1
        return walk

    @property
    def has_labels(self) -> bool:
        """
        Checks if graph nodes have labels.

        Returns:
            True if graph nodes have labels otherwise False
        """
        return self._labels is not None

    @property
    def labels(self) -> Dict[str, str]:
        """
        Gets graph node labels.

        Returns:
            Graph node labels
        """
        assert self._labels is not None, 'This dataset does not have any labels!'
        return self._labels



@register_dataset('graph_triplets')
class GraphTriplets(RandomWalkDataset):
    """
    Simple graph that consists of `NUM_CLUSTERS` fully connected triplets.
    This dataset is used as a sanity test.
    """
    NUM_CLUSTERS = 3

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
            method=method,
            labels=labels
        )


@register_dataset('graph_karate_club')
class KarateClubDataset(RandomWalkDataset):
    """
    Standard toy example for GNN tasks.
    """
    def __init__(self, walks_per_node: int, walk_length: int, method: str = 'deepwalk', **kwargs):
        graph = nx.karate_club_graph()
        mapping = {node: f'n{node + 1:02d}' for node in graph.nodes}  # Convert integers to strings
        graph = nx.relabel_nodes(graph, mapping)
        labels = {  # Reference: wikipedia
            'n01': '1', 'n02': '1', 'n03': '1', 'n04': '1', 'n05': '1',
            'n06': '1', 'n07': '1', 'n08': '1', 'n09': '1', 'n10': '2',
            'n11': '1', 'n12': '1', 'n13': '1', 'n14': '1', 'n15': '2',
            'n16': '2', 'n17': '1', 'n18': '1', 'n19': '2', 'n20': '1',
            'n21': '2', 'n22': '1', 'n23': '2', 'n24': '2', 'n25': '2',
            'n26': '2', 'n27': '2', 'n28': '2', 'n29': '2', 'n30': '2',
            'n31': '2', 'n32': '2', 'n33': '2', 'n34': '2'
        }

        super().__init__(
            graph=graph,
            walks_per_node=walks_per_node,
            walk_length=walk_length,
            method=method,
            labels=labels,
            **kwargs
        )
