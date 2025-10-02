import torch
from pathlib import Path
from typing import Literal
import tiktoken
from .config import batch_size, context_length, device

class TokenDataset:
    """
    Dataset that supports both character-level and BPE tokenization.
    
    Args:
        text: Input text to tokenize
        tokenizer_type: Either 'char' for character-level or 'bpe' for BPE tokenization
        bpe_encoding: Encoding name for tiktoken (e.g., 'gpt2', 'cl100k_base'). 
                     Only used if tokenizer_type='bpe'
    """
    def __init__(
        self, 
        text: str, 
        tokenizer_type: Literal['char', 'bpe'] = 'char',
        bpe_encoding: str = 'gpt2'
    ):
        self.tokenizer_type = tokenizer_type
        
        if tokenizer_type == 'char':
            self._init_char_tokenizer(text)
        elif tokenizer_type == 'bpe':
            self._init_bpe_tokenizer(text, bpe_encoding)
        else:
            raise ValueError(f"tokenizer_type must be 'char' or 'bpe', got {tokenizer_type}")
        
        # Encode full dataset and split
        data = self.encode(text)
        n = int(0.9 * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]
    
    def _init_char_tokenizer(self, text: str):
        """Initialize character-level tokenizer."""
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.tokenizer = None
        print(f"Character tokenizer initialized with vocab size: {self.vocab_size}")
    
    def _init_bpe_tokenizer(self, text: str, encoding_name: str):
        """Initialize BPE tokenizer using tiktoken."""
        try:
            self.tokenizer = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            raise ValueError(f"Failed to load BPE encoding '{encoding_name}': {e}")
        
        self.vocab_size = self.tokenizer.n_vocab
        self.stoi = None
        self.itos = None
        self.chars = None
        print(f"BPE tokenizer '{encoding_name}' initialized with vocab size: {self.vocab_size}")
    
    def encode(self, s: str) -> torch.Tensor:
        """Convert string to tensor of token indices."""
        if self.tokenizer_type == 'char':
            return torch.tensor([self.stoi[c] for c in s], dtype=torch.long)
        else:  # bpe
            tokens = self.tokenizer.encode(s)
            return torch.tensor(tokens, dtype=torch.long)
    
    def decode(self, t: torch.Tensor) -> str:
        """Convert tensor of token indices back to string."""
        if self.tokenizer_type == 'char':
            return ''.join(self.itos[int(i)] for i in t)
        else:  # bpe
            tokens = t.tolist() if isinstance(t, torch.Tensor) else t
            return self.tokenizer.decode(tokens)
    
    def get_batch(self, split: str, batch_size: int, context_length: int, device: str = 'cpu'):
        """
        Sample a mini-batch of sequences for training.
        
        Args:
            split: 'train' or 'val'
            batch_size: Number of sequences in batch
            context_length: Length of each sequence
            device: Device to put tensors on
        """
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

    
    
file_path = Path("data/crime-and-punishment.txt")

with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

print("=" * 60)
print("CHARACTER-LEVEL TOKENIZATION")
print("=" * 60)
char_dataset = TokenDataset(text, tokenizer_type='char')
print(f"Vocab size: {char_dataset.vocab_size}")
print(f"Train tokens: {len(char_dataset.train_data):,}")
print(f"Val tokens: {len(char_dataset.val_data):,}")

x, y = char_dataset.get_batch("train", batch_size, context_length, device)
print(f"Batch shape: x={x.shape}, y={y.shape}")
print(f"Sample decode: '{char_dataset.decode(x[0][:50])}'")

print("\n" + "=" * 60)
print("BPE TOKENIZATION (GPT-2)")
print("=" * 60)
bpe_dataset = TokenDataset(text, tokenizer_type='bpe', bpe_encoding='gpt2')
print(f"Vocab size: {bpe_dataset.vocab_size}")
print(f"Train tokens: {len(bpe_dataset.train_data):,}")
print(f"Val tokens: {len(bpe_dataset.val_data):,}")

x, y = bpe_dataset.get_batch("train", batch_size, context_length, device)
print(f"Batch shape: x={x.shape}, y={y.shape}")
print(f"Sample decode: '{bpe_dataset.decode(x[0][:50])}'")

# Comparison
print("\n" + "=" * 60)
print("COMPRESSION RATIO")
print("=" * 60)
print(f"Character tokens: {len(char_dataset.train_data) + len(char_dataset.val_data):,}")
print(f"BPE tokens: {len(bpe_dataset.train_data) + len(bpe_dataset.val_data):,}")
ratio = (len(char_dataset.train_data) + len(char_dataset.val_data)) / (len(bpe_dataset.train_data) + len(bpe_dataset.val_data))
print(f"Compression ratio: {ratio:.2f}x")