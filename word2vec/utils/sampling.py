import torch


def generate_noise_batch(batch_size: int, neg_samples: int, vocab_size: int):
    """
    Generates batch of noise words with shape (B, Ns) where B is batch size
    and Ns is number of negative samples per batch sample.

    Args:
        batch_size: Batch size
        neg_samples: Number of negative samples per batch sample
        vocab_size: Vocabulary size

    Returns:
        List of random words (sampled from uni-gram distributions)
    """
    return torch.randint(low=0, high=vocab_size, size=(batch_size, neg_samples), dtype=torch.long)