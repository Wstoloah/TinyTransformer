import torch 
from .config import *

text = '''From fairest creatures we desire increase,
That thereby beauty's rose might never die,
...'''

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for ch,i in stoi.items()}

encode = lambda s: torch.tensor([stoi[c] for c in s], dtype=torch.long)
decode = lambda t: ''.join(itos[int(i)] for i in t)


# Prepare train/validation split
data = encode(text)
n = int(0.9*len(data))
train_data, val_data = data[:n], data[n:]

# Function to sample mini-batches for training
def get_batch(split):
    d = train_data if split == 'train' else val_data
    # ensure we never sample beyond available length
    cur_block = min(context_length, max(2, len(d) - 2))
    hi = len(d) - cur_block - 1
    if hi <= 0:
        # fall back to using the whole sequence if extremely short
        x = d[:cur_block].unsqueeze(0)
        y = d[1:cur_block+1].unsqueeze(0)
        return x.to(device), y.to(device)

    ix = torch.randint(hi, (batch_size,))
    x = torch.stack([d[i:i+cur_block] for i in ix])
    y = torch.stack([d[i+1:i+cur_block+1] for i in ix])
    return x.to(device), y.to(device)