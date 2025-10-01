import os
import torch
from tqdm.auto import tqdm
from src.tinytransformer import TinyTransformer
from src.config import device, lr, max_iters, context_length
from src.utils import dataset

# Initialize the model
model = TinyTransformer().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# Training loop
print("Starting training...")
for it in tqdm(range(max_iters + 1)):
    # Get a batch of training data
    X_batch, y_batch = dataset.get_batch('train')

    # Forward pass
    logits, loss = model(X_batch, y_batch)

    # Backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # Print progress
    if it % 10 == 0:
        print(f"Iteration {it}/{max_iters}, Loss: {loss.item():.4f}")


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(project_root, "models")
os.makedirs(model_dir, exist_ok=True) 

model_path = os.path.join(model_dir, "trained_tiny_transformer_1.pth")

# Save
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
