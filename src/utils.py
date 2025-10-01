import torch
from .config import *

class CharDataset:
    def __init__(self, text: str):
        # Build vocabulary
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

        # Encode full dataset
        data = self.encode(text)
        n = int(0.9 * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def encode(self, s: str) -> torch.Tensor:
        """Convert string to tensor of indices."""
        return torch.tensor([self.stoi[c] for c in s], dtype=torch.long)

    def decode(self, t: torch.Tensor) -> str:
        """Convert tensor of indices back to string."""
        return ''.join(self.itos[int(i)] for i in t)

    def get_batch(self, split: str):
        """Sample a mini-batch of sequences for training."""
        d = self.train_data if split == 'train' else self.val_data

        # Handle very small datasets gracefully
        cur_block = min(context_length, len(d) - 1)
        if cur_block < 2:
            raise ValueError("Dataset too small for batching.")

        hi = len(d) - cur_block - 1
        if hi <= 0:
            # Use whole sequence as one batch
            x = d[:cur_block].unsqueeze(0)
            y = d[1:cur_block+1].unsqueeze(0)
            return x.to(device), y.to(device)

        # Random batch sampling
        ix = torch.randint(0, hi, (batch_size,))
        x = torch.stack([d[i:i+cur_block] for i in ix])
        y = torch.stack([d[i+1:i+cur_block+1] for i in ix])
        return x.to(device), y.to(device)


text = "I am ouissal :)"
dataset = CharDataset(text)