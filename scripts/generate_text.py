import argparse
import torch
from src.model import DecoderOnlyTransformer
from src.utils import setup_tokenizer, generate_text


def parse_args():
    parser = argparse.ArgumentParser(description='Generate text using a trained transformer model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--prompt', type=str, default="The", help='Text prompt to start generation')
    parser.add_argument('--max_length', type=int, default=100, help='Maximum length of generated text')
    parser.add_argument('--temperature', type=float, default=0.8, help='Temperature for sampling')

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize tokenizer
    tokenizer = setup_tokenizer()

    # Load model (you might need to adjust parameters to match your trained model)
    model = DecoderOnlyTransformer(
        vocab_size=50257,
        d_model=256,
        num_heads=4,
        num_layers=3,
        d_ff=1024,
        max_seq_len=256,
        dropout=0.2
    ).to(device)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Generate text
    generated = generate_text(model, tokenizer, args.prompt, args.max_length, args.temperature)
    print(f"Generated text: {generated}")


if __name__ == "__main__":
    main()