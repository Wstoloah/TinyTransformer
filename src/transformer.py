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

    def __init__(self, n_embd: int, dropout: float, hidden_mult: int = 4):
        """
        Initialize the MLP module.

        Args:
            n_embd (int): Embedding dimension.
            dropout (float): Dropout probability.
            hidden_mult (int): Multiplier for the hidden layer size. Default is 4.
        """
        super().__init__()
        hidden_size = hidden_mult * n_embd
        self.net = nn.Sequential(
            nn.Linear(n_embd, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, n_embd),
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
        return self.net(x)

class TransformerBlock(nn.Module):
    """
    Transformer block consisting of a causal self-attention layer and an MLP, with layer normalization and residual connections.

    Attributes:
        ln1, ln2 (nn.LayerNorm): Layer normalization layers applied before attention and MLP.
        attn (CausalSelfAttention): Causal self-attention module.
        mlp (MLP): Multi-layer perceptron module.
    """

    def __init__(self, n_embd: int, n_head: int, dropout: float, block_size: int):
        """
        Initialize the TransformerBlock module.

        Args:
            n_embd (int): Embedding dimension.
            n_head (int): Number of attention heads.
            dropout (float): Dropout probability.
            block_size (int): Maximum sequence length for the causal mask.
        """
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

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