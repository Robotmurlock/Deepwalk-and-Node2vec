import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from word2vec.dataloader import W2VDataset, W2VCollateFunctional
from word2vec.loss import NegativeSamplingLoss
from word2vec.model import SkipGram
from word2vec.utils import generate_noise_batch, pairwise_cosine_similarity


def train_model(
    model: nn.Module,
    loss_func: nn.Module,
    dataset: W2VDataset,
    solver_config: dict,
    dataloader_config: dict,
    collate_fn: W2VCollateFunctional,
    device: str
) -> None:
    # Trainer config
    max_epochs = solver_config['max_epochs']
    learning_rate = solver_config['lr']
    neg_samples = solver_config['neg_samples']

    # Dataloader config
    batch_size = dataloader_config['batch_size']
    num_workers = dataloader_config['num_workers']

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn, shuffle=True)
    vocab_size = len(dataset.vocab)

    optimizer = torch.optim.Adam(lr=learning_rate, params=model.parameters())

    model.train()
    model.to(device)
    loss_func.to(device)
    for epoch in range(1, max_epochs + 1):
        total_loss, total_positive_loss, total_negative_loss = 0.0, 0.0, 0.0
        n_steps = 0

        for inputs, outputs in tqdm(dataloader, unit='batch'):
            optimizer.zero_grad()

            inputs, outputs = inputs.to(device), outputs.to(device)
            noise = generate_noise_batch(inputs.shape[0], neg_samples, vocab_size).to(device)

            positive_logits = model(inputs, outputs, proba=False)
            negative_logits = model(inputs, noise, proba=False)

            loss, positive_loss, negative_loss = loss_func(positive_logits, negative_logits)
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().cpu().item()
            total_positive_loss += positive_loss.detach().cpu().item()
            total_negative_loss += negative_loss.detach().cpu().item()
            n_steps += 1

        avg_loss = total_loss / n_steps
        avg_positive_loss = total_positive_loss / n_steps
        avg_negative_loss = total_negative_loss / n_steps
        print(f'[Epoch={epoch}]: {avg_loss=}, {avg_positive_loss=}, {avg_negative_loss=}')


def main() -> None:
    dataset = W2VDataset(dataset_name='abcde', split='train', context_radius=1, min_word_frequency=1)
    model = SkipGram(vocab_size=len(dataset.vocab), embedding_size=2)

    train_model(
        model=model,
        loss_func=NegativeSamplingLoss(proba_input=False),
        dataset=dataset,
        solver_config={
            'max_epochs': 100,
            'lr': 1e-1,
            'neg_samples': 1,
        },
        dataloader_config={
            'num_workers': 8,
            'batch_size': 4
        },
        collate_fn=W2VCollateFunctional(mode='sg', context_radius=1, max_length=256),
        device='cuda:0'
    )

    print(dataset.vocab.get_stoi())
    inverse_map = {v: k for k, v in dataset.vocab.get_stoi().items()}
    input_emb = model.input_embedding
    output_emb = model.output_embedding
    sim = pairwise_cosine_similarity(input_emb, output_emb)
    print(sim)
    closest = torch.argmax(sim, dim=-1)
    closest = [inverse_map[int(x.item())] for x in closest]
    print(list(zip([inverse_map[i] for i in range(sim.shape[0])], closest)))



if __name__ == '__main__':
    main()