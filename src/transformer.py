import torch 
from torch import nn
from attention import CausalSelfAttention

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) module used in transformer blocks.

    Attributes:
        net (nn.Sequential): Sequential model consisting of two linear layers with a GELU activation in between,
                             and a dropout layer after the second linear layer.
    """

    def __init__(self, embed_dim: int, dropout: float, hidden_mult: int = 4):
        """
        Initialize the MLP module.

        Args:
            embed_dim (int): Embedding dimension.
            dropout (float): Dropout probability.
            hidden_mult (int): Multiplier for the hidden layer size. Default is 4.
        """
        super().__init__()
        hidden_size = hidden_mult * embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP module.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C).

        Returns:
            torch.Tensor: Output tensor of the same shape as input (B, T, C).
        """
        return self.mlp(x)

class TransformerBlock(nn.Module):
    """
    Transformer block consisting of a causal self-attention layer and an MLP, with layer normalization and residual connections.

    Attributes:
        ln1, ln2 (nn.LayerNorm): Layer normalization layers applied before attention and MLP.
        attn (CausalSelfAttention): Causal self-attention module.
        mlp (MLP): Multi-layer perceptron module.
    """

    def __init__(self, embed_dim: int, n_heads: int, dropout: float, context_length: int):
        """
        Initialize the TransformerBlock module.

        Args:
            embed_dim (int): Embedding dimension.
            n_heads (int): Number of attention heads.
            dropout (float): Dropout probability.
            context_length (int): Maximum sequence length for the causal mask.
        """
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, n_heads, dropout, context_length)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the TransformerBlock module.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C).

        Returns:
            torch.Tensor: Output tensor of the same shape as input (B, T, C).
        """
        # Apply layer normalization, attention, and residual connection
        x = x + self.attn(self.ln1(x))
        # Apply layer normalization, MLP, and residual connection
        x = x + self.mlp(self.ln2(x))
        return x