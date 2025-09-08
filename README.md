# Transformer Language Model

A PyTorch implementation of a decoder-only transformer for language modeling, trained on the WikiText-2 dataset.

## Features

- Multi-head self-attention mechanism
- Feed-forward networks with GELU activation
- Layer normalization and residual connections
- Causal masking for autoregressive generation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/KUSAI00/decoder_only_transformer.git
cd decoder-only-transformer

pip install -r requirements.txt
```

## Usage

### Training

To train the model with default parameters:

```bash
python main.py
```

Or use the training script:

```bash
python scripts/train_model.py --epochs 10 --batch_size 32 --lr 0.0001
```

### Text Generation

To generate text using a trained model:

```bash
python scripts/generate_text.py --model_path best_model.pth --prompt "The history of" --max_length 200
```

## Project Structure

```
decoder-only-transformer/
├── src/                 # Source code
├── scripts/             # Utility scripts
├── configs/             # Configuration files
├── requirements.txt     # Dependencies
└── README.md            # This file
```

## Results

The model was trained for **8 epochs** with early stopping.  
It achieved a **best validation loss of 2.0655** on the WikiText-2 test set.
