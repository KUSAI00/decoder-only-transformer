import argparse
import torch
from src.model import DecoderOnlyTransformer
from src.train import train_model
from src.utils import setup_tokenizer, load_wikitext_data, create_data_loaders


def parse_args():
    parser = argparse.ArgumentParser(description='Train a transformer language model')
    parser.add_argument('--max_seq_len', type=int, default=256, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of transformer layers')
    parser.add_argument('--d_ff', type=int, default=1024, help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize tokenizer and data
    tokenizer = setup_tokenizer()
    train_dataset, val_dataset, _ = load_wikitext_data(tokenizer, max_length=args.max_seq_len)
    train_loader, val_loader, _ = create_data_loaders(
        train_dataset, val_dataset, val_dataset, batch_size=args.batch_size
    )

    # Initialize model
    model = DecoderOnlyTransformer(
        vocab_size=50257,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout
    ).to(device)

    print(f"Model has {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    # Train the model
    train_model(
        model, train_loader, val_loader,
        num_epochs=args.epochs, learning_rate=args.lr, patience=2
    )


if __name__ == "__main__":
    main()