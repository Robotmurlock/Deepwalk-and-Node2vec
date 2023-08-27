"""
Random Walk Generator.
"""
import random
from abc import ABC, abstractmethod
from typing import List, Optional

import networkx as nx


class RandomWalk(ABC):
    """
    RandomWalk method interface definition.
    """
    def __init__(self, graph: nx.Graph, length: int):
        """
        Args:
            graph: Graph
            length: Random walk length
        """
        assert length >= 1, 'Minimum walk length is 1!'

        self._graph = graph
        self._length = length

    @abstractmethod
    def walk(self, node: str) -> str:
        """
        Performs a random walk starting from node `node`.
        Returns graph walk in sentence format.
        Example: `n1 n2 n3` for walk of nodes (n1, n2, n3)

        Args:
            node: Starting node

        Returns:
            walk as a sentence.
        """
        pass

    def get_node_neighbors(self, node: str) -> List[str]:
        return list(self._graph.neighbors(node))

    def get_node_unnormalized_edge_weights(self, node: str) -> List[float]:
        neighbors = list(self._graph.neighbors(node))
        if not nx.is_weighted(self._graph):
            return [1 for _ in neighbors]
        return [self._graph[node][neighbor]['weight'] for neighbor in neighbors]

    def get_node_normalized_edge_weights(self, node: str) -> List[float]:
        neighbor_weights = self.get_node_unnormalized_edge_weights(node)
        neighbor_weight_sum = sum(neighbor_weights)
        return [nw / neighbor_weight_sum for nw in neighbor_weights]


class DeepWalk(RandomWalk):
    """
    Implementation of simple random walk generator.
    Reference: https://arxiv.org/pdf/1403.6652.pdf
    """
    def walk(self, node: str) -> str:
        walk_nodes: List[str] = [node]

        while len(walk_nodes) < self._length:
            neighbors = self.get_node_neighbors(node)
            normalized_weights = self.get_node_normalized_edge_weights(node)

            child = random.choices(neighbors, weights=normalized_weights, k=1)[0]
            walk_nodes.append(child)
            node = child

        return ' '.join(walk_nodes)


class Node2Vec(RandomWalk):
    """
    Implementation of simple random walk generator.
    Reference: https://arxiv.org/pdf/1607.00653.pdf
    """
    def __init__(self, graph: nx.Graph, length: int, p: float = 1.0, q: float = 1.0):
        """
        Args:
            graph: Graph
            length: Random walk length
            p: Parameter p
        """
        super().__init__(
            graph=graph,
            length=length
        )
        self._p = p
        self._q = q

    def walk(self, node: str) -> str:
        walk_nodes: List[str] = [node]

        prev_node = None
        while len(walk_nodes) < self._length:
            neighbors = self.get_node_neighbors(node)
            neighbor_weights = self.get_node_unnormalized_edge_weights(node)
            for i, candidate_child in enumerate(neighbors):
                if candidate_child == prev_node:  # shortest path length between "next" and previous node is 0
                    neighbor_weights[i] *= 1 / self._p
                    continue

                candidate_neighbors = self.get_node_neighbors(candidate_child)
                if prev_node in candidate_neighbors:  # shortest path length between "next" and previous node is 1
                    neighbor_weights[i] *= 1 / self._q

            neighbor_weight_sum = sum(neighbor_weights)
            normalized_weights = [nw / neighbor_weight_sum for nw in neighbor_weights]

            child = random.choices(neighbors, weights=normalized_weights, k=1)[0]
            walk_nodes.append(child)

            prev_node = node
            node = child

        return ' '.join(walk_nodes)


def random_walk_factory(name: str, graph: nx.Graph, length: int, additional_params: Optional[dict] = None) -> RandomWalk:
    """
    Creates random walk method object.

    Args:
        name: Method name
        graph: Graph
        length: Walk length
        additional_params: Additional method specific parameters
    Returns:
        RandomWalk generator.
    """
    name = name.lower()
    if additional_params is None:
        additional_params = {}

    SUPPORTED_METHODS = {
        'deepwalk': DeepWalk,
        'dfs': DeepWalk,
        'node2vec': Node2Vec
    }

    assert name in SUPPORTED_METHODS, f'Unknown method "{name}". Supported: {list(SUPPORTED_METHODS.keys())}'


    return SUPPORTED_METHODS[name](
        graph=graph,
        length=length,
        **additional_params
    )