from torch.utils.data import Dataset
from tqdm import tqdm
import torch


class WikiTextDataset(Dataset):
    def __init__(self, tokenizer, texts, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.sequences = self._process_data(texts)
        print(f"Number of sequences: {len(self.sequences)}")

    def _process_data(self, texts):
        sequences = []
        for text in tqdm(texts, desc="Processing data"):
            text = text.strip()
            if len(text) < 10:
                continue

            tokens = self.tokenizer.encode(text)

            # Pad or split the sequence
            if len(tokens) < self.max_length:
                # Pad to max_length for consistency.
                padded = tokens + [self.tokenizer.eos_token_id] * (self.max_length - len(tokens))
                sequences.append(padded)
            else:
                for j in range(0, len(tokens) - self.max_length, self.max_length // 2):
                    chunk = tokens[j:j + self.max_length]
                    sequences.append(chunk)

        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        input_ids = torch.tensor(seq[:-1], dtype=torch.long)
        targets = torch.tensor(seq[1:], dtype=torch.long)
        return input_ids, targets
