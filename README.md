# TinyTransformer

TinyTransformer is a lightweight implementation of a GPT like transforler model built with PyTorch. It is designed for educational purposes and small-scale sequence modeling tasks, such as text generation.

## Features
- **Causal Self-Attention**: Implements autoregressive attention for sequence generation.
- **Configurable Hyperparameters**: Easily adjust model size, number of layers, and more.
- **Training and Generation Scripts**: Includes scripts for training the model and generating text.
- **Modular Design**: Clean and extensible codebase.

## Project Structure
```
TinyTransformer/
│
├── src/                        # Source code for the project
│   ├── attention.py            # Attention mechanism implementation
│   ├── config.py               # Configuration parameters
│   ├── transformer.py          # Transformer blocks and MLP
│   ├── main.py                 # TinyTransformer model definition
│   ├── datasets.py             # Dataset creation with character level and BPE tokenization
│   └── __init__.py             # Makes `src` a package
│
├── scripts/                    # Standalone scripts for specific tasks
│   ├── train.py                # Script for training the model
│   ├── generate.py             # Script for generating text
│   └── __init__.py             # Makes `scripts` a package
│
├── data/                       # Data used for training
│
├── models/                     # Saved models and checkpoints
│
├── requirements.txt            # List of dependencies
├── README.md                   # Project overview and instructions
├── .gitignore                  # Git ignore file
└── setup.py                    # For packaging the project
```

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Wstoloah/TinyTransformer.git
   cd TinyTransformer
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Training the Model
To train the model, run the following command:
```bash
python -m scripts.train
```

### Generating Text
To generate text using a trained model, run:
```bash
python -m scripts.generate
```

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- Inspired by the Transformer architectures introduced in the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762) and the paper [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- Built with PyTorch.