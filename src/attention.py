import math
from torch import nn
import torch
import torch.nn.functional as F  # Corrected import for clarity

from config import *

class CausalSelfAttention(nn.Module):
    """
    Causal Self-Attention module with support for multiple attention heads and dropout.

    Attributes:
        n_head (int): Number of attention heads.
        key, query, value, proj (nn.Linear): Linear layers for computing key, query, value, and output projections.
        attn_drop (nn.Dropout): Dropout applied to attention weights.
        resid_drop (nn.Dropout): Dropout applied after the final projection.
        mask (torch.Tensor): Causal mask to prevent attention to future tokens.
    """

    def __init__(self, n_embd: int, n_head: int, dropout: float, block_size: int):
        """
        Initialize the CausalSelfAttention module.

        Args:
            n_embd (int): Embedding dimension.
            n_head (int): Number of attention heads.
            dropout (float): Dropout probability.
            block_size (int): Maximum sequence length for the causal mask.
        """
        super().__init__()
        self.n_head = n_head
        self.block_size = block_size

        # Projections: these will later be split into n_head smaller heads.
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_drop = nn.Dropout(dropout)  # Applied to attention weights
        self.resid_drop = nn.Dropout(dropout)  # Applied after the final projection

        # Causal mask: prevents looking ahead
        self.register_buffer(
            'mask',
            torch.tril(torch.ones(block_size, block_size))
            .view(1, 1, block_size, block_size),
            persistent=False  # Mask is not trainable and not saved with the model
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CausalSelfAttention module.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C), where B is the batch size,
                              T is the sequence length, and C is the embedding dimension.

        Returns:
            torch.Tensor: Output tensor of the same shape as input (B, T, C).
        """
        B, T, C = x.size()

        if T > self.block_size:
            raise ValueError(f"Sequence length T ({T}) cannot exceed block_size ({self.block_size}).")

        # Compute key, query, value projections and reshape into multiple heads
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        # Compute attention output
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))