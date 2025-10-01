import torch
from src.tinytransformer import TinyTransformer
from src.config import device
from src.utils import dataset

if __name__ == "__main__":
    # Load the trained model
    model = TinyTransformer().to(device)
    model_path = "models/trained_tiny_transformer_crime-and-punishment.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("**TinyTransformer Text Generator**")
    user_prompt = input("Enter some text to start generation: ").lower()

    # Encode user input into tokens
    context = dataset.encode(user_prompt).unsqueeze(0).to(device)

    # Generate new tokens
    with torch.no_grad():
        sample = model.generate(context, max_new_tokens=200)[0].tolist()

    # Decode tokens back into text
    generated_text = dataset.decode(sample)

    print("\n--- Generated text ---")
    print(generated_text)
    print("----------------------")
