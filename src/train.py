import torch.nn as nn
from tqdm import tqdm
import torch
import os


def train_model(model, train_loader, val_loader, device, num_epochs=15, learning_rate=3e-4, patience=2):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')
        for batch_idx, (input_ids, targets) in enumerate(train_pbar):
            input_ids, targets = input_ids.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Val]')
            for input_ids, targets in val_pbar:
                input_ids, targets = input_ids.to(device), targets.to(device)
                logits = model(input_ids)
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                total_val_loss += loss.item()
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f'Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}')

        # Early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"✅ New best model saved with validation loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(f"⚠️ No improvement in validation loss for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print(f"Stopping early after {epoch + 1} epochs.")
            break

    return train_losses, val_losses


def evaluate_model(model, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc='Testing')
        for input_ids, targets in test_pbar:
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()
            test_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    print(f'Test Loss: {avg_loss:.4f}')
    print(f'Test Perplexity: {perplexity:.2f}')

    return avg_loss, perplexity


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
