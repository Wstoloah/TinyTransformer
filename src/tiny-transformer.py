from config import *
from utils import vocab_size
from utils import get_batch, decode
import torch
from torch import nn
from torch.nn.functional import F
from transformer import TransformerBlock

class TinyTransformer(nn.Module):
    """
    A tiny transformer model for sequence modeling tasks, including token embedding,
    positional embedding, transformer blocks, and a final linear layer for predictions.

    Attributes:
        tok_emb (nn.Embedding): Token embedding layer.
        pos_emb (nn.Embedding): Positional embedding layer.
        drop (nn.Dropout): Dropout layer applied after embeddings.
        blocks (nn.Sequential): Sequence of transformer blocks.
        ln_f (nn.LayerNorm): Final layer normalization.
        head (nn.Linear): Linear layer for output logits.
    """

    def __init__(self):
        """
        Initialize the TinyTransformer model.

        Args:
            None (uses global config parameters).
        """
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(context_length, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, n_heads, dropout, context_length) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Tie weights: reuse embedding weights in output
        self.head.weight = self.tok_emb.weight

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass for the TinyTransformer model.

        Args:
            idx (torch.Tensor): Input tensor of token indices (batch_size, seq_length).
            targets (torch.Tensor, optional): Target tensor for loss computation (batch_size, seq_length).

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]:
                - logits (torch.Tensor): Output logits of shape (batch_size, seq_length, context_length).
                - loss (torch.Tensor | None): Cross-entropy loss if targets are provided.
        """
        batch_size, seq_len = idx.shape

        # Token and positional embeddings
        tok = self.tok_emb(idx)  # (batch_size, seq_length, embed_dim)
        pos = self.pos_emb(torch.arange(seq_len, device=idx.device))  # (seq_len, embed_dim)
        x = self.drop(tok + pos)  # Add embeddings and apply dropout

        # Transformer blocks
        x = self.blocks(x)
        x = self.ln_f(x)  # Final layer normalization

        # Compute logits
        logits = self.head(x)  # (batch_size, seq_len, context_length)

        # Compute loss if targets are provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        Generate new tokens autoregressively.

        Args:
            idx (torch.Tensor): Input tensor of token indices (batch, seq_len).
            max_new_tokens (int): Maximum number of new tokens to generate.

        Returns:
            torch.Tensor: Generated sequence of token indices (B, T + max_new_tokens).
        """
        for _ in range(max_new_tokens):
            # Crop context to the last block_size tokens
            idx_cond = idx[:, -context_length:]

            # Forward pass to get logits
            logits, _ = self(idx_cond)

            # Focus only on the last token's logits
            logits = logits[:, -1, :]  # (batch_size, context_length)

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample from the distribution
            next_id = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

            # Append sampled token to the sequence
            idx = torch.cat((idx, next_id), dim=1)  # (batch_size, seq_len+1)

        return idx
