import math
from torch import nn
import torch
import torch.nn.functional as F
from .config import *

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

    def __init__(self, embed_dim: int, n_head: int, dropout: float, context_length: int):
        """
        Initialize the CausalSelfAttention module.

        Args:
            embed_dim (int): Embedding dimension.
            n_head (int): Number of attention heads.
            dropout (float): Dropout probability.
            context_length (int): Maximum sequence length for the causal mask.
        """
        super().__init__()
        assert embed_dim % n_head == 0, "embed_dim must be divisible by num_heads"
        self.n_head = n_head
        self.context_length = context_length

        # Projections: these will later be split into n_head smaller heads.
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

        # Linear layer to combine head outputs
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False) 

        self.attn_drop = nn.Dropout(dropout)  # Applied to attention weights
        self.resid_drop = nn.Dropout(dropout)  # Applied after the final projection

        # Causal mask: prevents looking ahead
        self.register_buffer(
            'mask',
            torch.tril(torch.ones(context_length, context_length))
            .view(1, 1, context_length, context_length),
            persistent=False  # Mask is not trainable and not saved with the model
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CausalSelfAttention module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim), where batch_size is the batch size,
                      seq_len is the sequence length, and embed_dim is the embedding dimension.

        Returns:
            torch.Tensor: Output tensor of the same shape as input (batch_size, seq_len, embed_dim).
        """
        batch_size, seq_len, embed_dim = x.size()

        if seq_len > self.context_length:
            raise ValueError(f"Sequence length  ({seq_len}) cannot exceed context_length ({self.context_length}).")

        # Compute key, query, value projections and reshape into multiple heads
        k = self.key(x).view(batch_size, seq_len, self.n_head, embed_dim // self.n_head).transpose(1, 2)
        q = self.query(x).view(batch_size, seq_len, self.n_head, embed_dim // self.n_head).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.n_head, embed_dim // self.n_head).transpose(1, 2)

        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.mask[:, :, :seq_len, :seq_len] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        # Compute attention output
        context_vec = att @ v
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        context_vec = self.resid_drop(self.proj(context_vec))
        return context_vec
