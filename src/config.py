import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

context_length   = 128   # how many characters the model looks back
batch_size   = 32    # sequences per batch
n_layer      = 2     # number of transformer blocks
n_heads       = 2     # attention heads
embed_dim    = 128   # embedding size
dropout      = 0.1
max_iters    = 1200  # training steps
lr           = 3e-3