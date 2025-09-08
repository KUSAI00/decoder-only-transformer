import torch
import matplotlib.pyplot as plt
from src.model import DecoderOnlyTransformer
from src.train import train_model, evaluate_model
from src.utils import setup_tokenizer, load_wikitext_data, create_data_loaders
from src.utils import generate_text

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
vocab_size = 50257
max_seq_len = 256
batch_size = 16
num_epochs = 15
learning_rate = 3e-4

# Model configuration
d_model = 256
num_heads = 4
num_layers = 3
d_ff = 1024
dropout = 0.2


def main():
    # Initialize tokenizer
    print("Loading tokenizer...")
    tokenizer = setup_tokenizer()

    # Load and process dataset
    print("Loading and processing WikiText-2 dataset...")
    train_dataset, val_dataset, test_dataset = load_wikitext_data(tokenizer, max_length=max_seq_len)
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset, batch_size=batch_size
    )

    # Initialize model
    print("Initializing model...")
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=dropout
    ).to(device)

    print(f"Model has {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    # Train the model
    print("Starting training...")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, device,
        num_epochs=num_epochs, learning_rate=learning_rate, patience=2
    )

    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_curves.png')
    plt.show()

    # Load the best model before evaluating
    try:
        print("\nLoading best model for evaluation...")
        model.load_state_dict(torch.load('best_model.pth'))
        print("Best model loaded.")
    except FileNotFoundError:
        print("\nWarning: No best model found. Evaluating final epoch model.")

    # Evaluate on test set
    print("Evaluating on test set...")
    test_loss, test_perplexity = evaluate_model(model, test_loader)

    # Generate some sample text
    print("\nGenerating sample text:")
    prompts = ["The history of", "In the year", "Scientists have discovered"]

    for prompt in prompts:
        generated = generate_text(model, tokenizer, prompt, max_length=50)
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: {generated}")

    print("\nScript finished.")


if __name__ == "__main__":
    main()