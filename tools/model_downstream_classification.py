"""
Performs downstream task on dataset labels.
Note: Dataset has to support labels.
Note: This is specialized for graph shallow encoders (deepwalk/node2vec)
"""
import logging
import os
import random
from pathlib import Path
from typing import List, Tuple

import hydra
import matplotlib.pyplot as plt
import networkx as nx
import numpy
import numpy as np
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression

from shallow_encoders.common.path import CONFIG_PATH
from shallow_encoders.graph import edge_operators
from shallow_encoders.word2vec.dataloader.torch_dataset import W2VDataset
from shallow_encoders.word2vec.model import W2VBase
from tools import conventions
from tools.utils import setup_pipeline, MATPLOTLIB_COLORS

logger = logging.getLogger('DownstreamTask-Classification')


def labels_to_integers(labels: List[str]) -> List[int]:
    """
    Convert a list of unique string labels to integers.

    Args:
        labels: List of string labels.

    Returns:
        List of integer representations of the labels.
    """
    label_to_int = {label: idx for idx, label in enumerate(set(labels))}
    int_labels = [label_to_int[label] for label in labels]
    return int_labels


def plot_logistic_regression_decision_boundary_line(points: np.ndarray, clf: LogisticRegression) -> None:
    """
    Plots logistic boundary lines.

    Args:
        points: 2D points
        clf: Classifier model
    """
    n_lines = clf.coef_.shape[0]

    for i in range(n_lines):
        theta0 = clf.intercept_[i]
        theta1, theta2 = clf.coef_[i, :]
        x_values = np.linspace(points[:, 0].min() - 1, points[:, 0].max() + 1, 100)
        y_values = (-theta0 - theta1 * x_values) / theta2

        plt.plot(x_values, y_values, color='red', label=f'Decision Boundary {i:03d}')


def create_and_fit_classification_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X: np.ndarray,
    y: np.ndarray
) -> Tuple[LogisticRegression, float]:
    """
    Creates, fits and evaluates linear logistic regression model to given data.

    Args:
        X_train: Train input data
        y_train: Train labels
        X: Prediction input data
        y: Prediction labels

    Returns:
        Classifier, accuracy
    """
    clf = LogisticRegression(max_iter=1_000)
    clf.fit(X_train, y_train)

    # Evaluate model
    y_hat = clf.predict(X)
    return clf, float(np.equal(y_hat, y).astype(np.float32).mean())


def perform_node_classification(
    model: W2VBase,
    dataset: W2VDataset,
    output_path: str,
    train_ratio: float,
    n_experiments: int,
    visualize: bool
) -> None:
    """
    Performs node classification downstream tasks where input features
    are obtained using random walk graph shallow encoders.

    Args:
        model: Graph Shallow Encoder
        dataset: Dataset
        output_path: Output path
        train_ratio: How much of data is used to train a linear classifier
        n_experiments: Number of experiments to perform
        visualize: Visualize best model
    """
    X = model.input_embedding.numpy()[1:, :]  # Skip `<unk>`
    vertices = dataset.vocab.get_itos()[1:]  # Skip `<unk>`
    vertex_labels = [dataset.labels[v] for v in vertices]
    y = np.array(labels_to_integers(vertex_labels), dtype=np.float32)

    n_samples = y.shape[0]
    n_train_samples = round(train_ratio * n_samples)

    best_accuracy, best_clf = None, None
    accuracy_sum = 0.0
    for i in range(n_experiments):
        # Create train dataset
        indices = list(range(n_samples))
        random.shuffle(indices)
        sampled_indices = sorted(indices[:n_train_samples])
        X_train = X[sampled_indices, :].copy()
        y_train = y[sampled_indices].copy()

        clf, accuracy = create_and_fit_classification_model(X_train, y_train, X, y)
        accuracy_sum += accuracy

        if best_accuracy is None or accuracy >= best_accuracy:
            best_accuracy = accuracy
            best_clf = clf

    assert best_clf is not None, 'No experiments performed!'
    accuracy = accuracy_sum / n_experiments
    logger.info(f'Node classification accuracy: {100 * accuracy:.2f}% (averaged over {n_experiments} experiments).')
    logger.info(f'Best accuracy score: {100 * best_accuracy:.2f}%.')

    # Plot (best)
    if visualize:
        fig = plt.figure(figsize=(10, 10))
        unique_labels = set(dataset.labels.values())
        for label, color in zip(unique_labels, MATPLOTLIB_COLORS):
            label_indices = [i for i, vl in enumerate(vertex_labels) if vl == label]
            X_label = X[label_indices, :]
            plt.scatter(X_label[:, 0], X_label[:, 1], color=color, label=label)

        plot_logistic_regression_decision_boundary_line(X, best_clf)
        plt.title(f'Classification on embeddings with {100 * train_ratio:.2f}% known data - Accuracy {100 * best_accuracy:.2f}')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.legend()

        fig_path = os.path.join(output_path, 'downstream-node-classification.jpg')
        fig.savefig(fig_path)
        logger.info(f'Saved figure at path "{fig_path}".')


def sample_negative_edges(graph: nx.Graph, n: int) -> List[Tuple[str, str]]:
    """
    Sample `n` negative sample edges.
    This algorithm may include duplicate negative samples,
    but it's rare for medium and large sized graphs

    Args:
        graph: Graph
        n: Size

    Returns:
        Sample of negative edges (non-existent).
    """
    nodes = list(graph.nodes)

    sampled_negative_edges: List[Tuple[str, str]] = []
    for _ in range(n):
        is_feasible_sample = False

        while not is_feasible_sample:
            node = random.choice(nodes)
            neighbors = list(graph.neighbors(node))
            not_neighbors = list(set(nodes) - set(neighbors))
            if len(not_neighbors) >= 1:
                # There is at least one random edge to choose
                other_node = random.choice(not_neighbors)
                sampled_negative_edges.append((node, other_node))

                is_feasible_sample = True

    return sampled_negative_edges


def create_edge_embeddings(
    node_embeddings: np.ndarray,
    edges: List[Tuple[int, int]],
    edge_operator: edge_operators.EdgeOperator
) -> np.ndarray:
    """
    Creates edge embedding data for training a linear classifier.

    Args:
        node_embeddings: Node embeddings
        edges: List of edges to create embeddings for
        edge_operator: Edge operator (e.g. Hadamard)

    Returns:
        Edge embeddings
    """
    edge_embeddings: List[np.ndarray] = []
    for s, e in edges:
        edge_embedding = edge_operator(node_embeddings[s, :], node_embeddings[e, :])
        edge_embeddings.append(edge_embedding)

    return np.stack(edge_embeddings)


def perform_edge_classification(
    model: W2VBase,
    dataset: W2VDataset,
    train_ratio: float,
    n_experiments: int,
    edge_operator_name: str
) -> None:
    """
    Performs edge classification for given model on given dataset.

    Args:
        model: Shallow graph encoder network.
        dataset: Dataset
        train_ratio:
        n_experiments:
        edge_operator_name:

    Returns:

    """
    node_embeddings = model.input_embedding.numpy()

    graph = dataset.graph
    token_to_index = dataset.vocab.get_stoi()
    edges = list(graph.edges)
    n_edges = len(edges)
    edge_operator = edge_operators.edge_operator_factory(edge_operator_name)

    best_accuracy = None
    accuracy_sum = 0.0
    for i in range(n_experiments):
        # Create train dataset
        n_train_samples = round(train_ratio * n_edges)
        n_val_samples = n_edges - n_train_samples

        # Sample positive samples
        random.shuffle(edges)
        train_positive_edges = edges[:n_train_samples]

        # Sample negative samples
        train_negative_edges = sample_negative_edges(graph, n_train_samples)
        val_negative_edges = sample_negative_edges(graph, n_val_samples)

        # Create dataset for LR
        train_edges = train_positive_edges + train_negative_edges
        train_edges = [(token_to_index[s], token_to_index[e]) for s, e in train_edges]
        y_train = np.array(n_train_samples * [1] + n_train_samples * [0], dtype=np.float32)
        all_edges = edges + train_negative_edges + val_negative_edges
        all_edges = [(token_to_index[s], token_to_index[e]) for s, e in all_edges]
        y = numpy.array(n_edges * [1] + n_train_samples * [0] + n_val_samples * [0], dtype=np.float32)

        X_train = create_edge_embeddings(
            node_embeddings=node_embeddings,
            edges=train_edges,
            edge_operator=edge_operator
        )

        X = create_edge_embeddings(
            node_embeddings=node_embeddings,
            edges=all_edges,
            edge_operator=edge_operator
        )

        _, accuracy = create_and_fit_classification_model(X_train, y_train, X, y)
        accuracy_sum += accuracy

        if best_accuracy is None or accuracy >= best_accuracy:
            best_accuracy = accuracy

    assert best_accuracy is not None, 'No experiments performed!'
    accuracy = accuracy_sum / n_experiments
    logger.info(f'Edge classification accuracy: {100 * accuracy:.2f}% (averaged over {n_experiments} experiments).')
    logger.info(f'Best accuracy score: {100 * best_accuracy:.2f}%.')


@hydra.main(config_path=CONFIG_PATH, config_name='w2v_sg_graph_triplets.yaml')
def main(cfg: DictConfig) -> None:
    cfg = setup_pipeline(cfg, task='downstream-classification')
    dataset = cfg.datamodule.instantiate_dataset()
    checkpoint_path = conventions.get_checkpoint_path(cfg.path.output_dir, cfg.datamodule.dataset_name,
                                                      cfg.train.experiment, cfg.analysis.checkpoint)
    pl_trainer = cfg.instantiate_trainer(dataset=dataset, checkpoint_path=checkpoint_path)

    analysis_exp_path = \
        conventions.get_analysis_experiment_path(cfg.path.output_dir, cfg.datamodule.dataset_name, cfg.train.experiment)
    Path(analysis_exp_path).mkdir(parents=True, exist_ok=True)

    if cfg.downstream.node_classification.enable:
        perform_node_classification(
            model=pl_trainer.model,
            dataset=dataset,
            output_path=analysis_exp_path,
            train_ratio=cfg.downstream.node_classification.train_ratio,
            n_experiments=cfg.downstream.node_classification.n_experiments,
            visualize=cfg.downstream.node_classification.visualize
        )

    if cfg.downstream.edge_classification.enable:
        perform_edge_classification(
            model=pl_trainer.model,
            dataset=dataset,
            edge_operator_name=cfg.downstream.edge_classification.operator_name,
            train_ratio=cfg.downstream.edge_classification.train_ratio,
            n_experiments=cfg.downstream.edge_classification.n_experiments
        )


if __name__ == '__main__':
    main()
