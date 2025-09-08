from transformers import GPT2TokenizerFast
from datasets import load_dataset
from torch.utils.data import DataLoader
from src.data_utils import WikiTextDataset


def setup_tokenizer():
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_wikitext_data(tokenizer, max_length=256):
    train_texts_raw = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')['text']
    val_texts_raw = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')['text']
    test_texts_raw = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')['text']

    train_dataset = WikiTextDataset(tokenizer, train_texts_raw, max_length=max_length)
    val_dataset = WikiTextDataset(tokenizer, val_texts_raw, max_length=max_length)
    test_dataset = WikiTextDataset(tokenizer, test_texts_raw, max_length=max_length)

    return train_dataset, val_dataset, test_dataset


def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=16):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def generate_text(model, tokenizer, prompt="The", max_length=100, temperature=0.8):
    model.eval()
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)

    with torch.no_grad():
        for _ in range(max_length):
            # Only use the last part of the sequence if it exceeds max_seq_len
            current_input = input_ids if input_ids.size(1) <= model.max_seq_len else input_ids[:, -model.max_seq_len:]

            logits = model(current_input)
            next_token_logits = logits[0, -1, :] / temperature
            next_token_probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(next_token_probs, 1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    generated_text = tokenizer.decode(input_ids[0].cpu().tolist())
    return generated_text
