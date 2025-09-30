# # Generate text
# print("Generating text...")
# context = torch.zeros((1, 1), dtype=torch.long, device=device)  # Start with an empty context
# sample = model.generate(context, max_new_tokens=400)[0].tolist()
# print("Generated text:")
# print(decode(sample))

# if __name__ == "__main__":
#     import torch

#     # Initialize the model
#     model = TinyTransformer()

#     # Example input: batch of token indices (B=2, T=5)
#     example_input = torch.randint(0, context_length, (2, 5))

#     # Forward pass
#     logits, loss = model(example_input, targets=example_input)

#     # Print the logits and loss
#     print("Logits:", logits)
#     if loss is not None:
#         print("Loss:", loss.item())
#     # Generate new tokens
#     generated = model.generate(example_input, max_new_tokens=5)
#     print("Generated sequence:", generated)