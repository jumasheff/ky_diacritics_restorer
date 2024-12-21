import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import KyrgyzTextDataset, DiacriticsRestorer
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import os

def train(model, train_dataset, val_dataset=None, epochs=10, batch_size=32, learning_rate=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.pad_idx)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    if val_dataset:
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch_idx, (src, tgt) in enumerate(progress_bar):
            src, tgt = src.to(device), tgt.to(device)
            
            optimizer.zero_grad()
            output = model(src)
            
            loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        if val_dataset:
            model.eval()
            total_val_loss = 0
            
            with torch.no_grad():
                for src, tgt in val_loader:
                    src, tgt = src.to(device), tgt.to(device)
                    output = model(src)
                    loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))
                    total_val_loss += loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}')
        else:
            print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}')
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    if val_losses:
        plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.show()
    
    return train_losses, val_losses

def test_model(model, dataset, text_samples):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print("\nTesting model on sample inputs:")
    for text in text_samples:
        restored = restore_diacritics(model, text, dataset, device)
        print(f"Input:  {text}")
        print(f"Output: {restored}\n")

if __name__ == "__main__":
    # Example usage
    EPOCHS = 10
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    
    # Load dataset
    dataset = KyrgyzTextDataset('example_dataset.tsv')
    
    # Create model
    model = DiacriticsRestorer(
        vocab_size=len(dataset.char_to_idx),
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        max_len=512
    )
    
    # Train model
    train_losses, _ = train(
        model=model,
        train_dataset=dataset,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE
    )
    
    # Test samples
    test_samples = [
        "кыргызстан онугуп кетет",
        "мен уйронуп жатам",
        "биз омур бою окууга даярбыз"
    ]
    
    test_model(model, dataset, test_samples)
