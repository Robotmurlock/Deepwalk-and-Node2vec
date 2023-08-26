import logging
import os
import random
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig
from sklearn.manifold import TSNE

from tools import conventions
from common.path import CONFIG_PATH
from tools.utils import setup_pipeline
from word2vec.dataloader import W2VDataset
from word2vec.model import W2VBase
from word2vec.utils.func import pairwise_cosine_similarity

logger = logging.getLogger('ModelAnalysis')


def show_closest_pairs_for_each_word(
    model: W2VBase,
    dataset: W2VDataset,
    output_path: str,
    max_words: int = 100,
    pairs_per_word: int = 5
) -> None:
    input_emb = model.input_embedding
    output_emb = model.output_embedding
    inverse_map = {v: k for k, v in dataset.vocab.get_stoi().items()}

    # Sample words for presentation
    vocab_size = len(dataset.vocab)
    if vocab_size > max_words:
        _, sampled_indices = dataset.get_n_most_frequent_words(max_words)
        input_emb = input_emb[sampled_indices]
    else:
        sampled_indices = list(range(vocab_size))

    # Perform cosine similarity
    sim = pairwise_cosine_similarity(input_emb, output_emb)

    text = [r'Closest pairs in format "{word}:{closest_word_pairs}"']
    for sim_index, word_index in enumerate(sampled_indices):
        word = inverse_map[word_index]
        closest_pair_indices = torch.argsort(sim[sim_index, :], descending=True)[:pairs_per_word]
        closest_pairs = [inverse_map[int(x.item())] for x in closest_pair_indices]
        closest_pairs_str = ', '.join(closest_pairs)
        text.append(f'{word}: {closest_pairs_str}')

    text = '\n'.join(text)
    logger.info(text)

    result_path = os.path.join(output_path, 'closest_pairs.txt')
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write(text)

    logger.info(f'Saved closest pairs analysis result at path "{result_path}".')

def visualize_embeddings(
    model: W2VBase,
    dataset: W2VDataset,
    output_path: str,
    max_words: int
):
    embeddings = model.input_embedding.numpy()
    words = dataset.vocab.get_itos()

    # Sample words for presentation
    vocab_size = len(dataset.vocab)
    if vocab_size > max_words:
        _, sampled_indices = dataset.get_n_most_frequent_words(max_words)
        embeddings = embeddings[sampled_indices]
    else:
        sampled_indices = list(range(vocab_size))
    words = [words[i] for i in sampled_indices]

    # Assert the embedding dimension is 2 or larger
    n_dims = embeddings.shape[1]
    assert n_dims >= 2, "Embedding dimension should be 2 or larger."

    # Use T-SNE for dimensionality reduction if necessary
    if n_dims > 2:
        tsne = TSNE(n_components=2, random_state=42)
        embeddings = tsne.fit_transform(embeddings)

    # Create the visualization
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.6)

    # Annotate the words on the visualization
    for i, word in enumerate(words):
        plt.annotate(word, (embeddings[i, 0], embeddings[i, 1]))

    plt.title("Word Embeddings Visualization")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)

    # Save the figure
    save_path = os.path.join(output_path, 'projected_embeddings.jpg')
    fig.savefig(save_path)
    plt.close(fig)

    logger.info(f'Saved embedding visualization at path "{save_path}".')


@hydra.main(config_path=CONFIG_PATH, config_name='w2v_sg_abcde.yaml')
def main(cfg: DictConfig) -> None:
    cfg = setup_pipeline(cfg, task='analysis')
    dataset = cfg.datamodule.instantiate_dataset()
    checkpoint_path = conventions.get_checkpoint_path(cfg.path.output_dir, cfg.datamodule.dataset_name,
                                                      cfg.train.experiment, cfg.analysis.checkpoint)
    pl_trainer = cfg.instantiate_trainer(dataset=dataset, checkpoint_path=checkpoint_path)

    analysis_exp_path = \
        conventions.get_analysis_experiment_path(cfg.path.output_dir, cfg.datamodule.dataset_name, cfg.train.experiment)
    Path(analysis_exp_path).mkdir(parents=True, exist_ok=True)

    if cfg.analysis.closest_pairs:
        logger.info('Performing closest pairs analysis...')
        show_closest_pairs_for_each_word(
            model=pl_trainer.model,
            dataset=dataset,
            max_words=cfg.analysis.closest_max_words,
            pairs_per_word=cfg.analysis.closest_pairs_per_word,
            output_path=analysis_exp_path
        )

    if cfg.analysis.visualize_embeddings:
        logger.info('Performing embedding visualization...')
        visualize_embeddings(
            model=pl_trainer.model,
            dataset=dataset,
            output_path=analysis_exp_path,
            max_words=cfg.analysis.visualize_embeddings_max_words
        )


if __name__ == '__main__':
    main()
