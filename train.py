import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import KyrgyzTextDataset, DiacriticsRestorer, restore_diacritics
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import os

def train(model, dataset, epochs=10, batch_size=32, learning_rate=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.pad_idx)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Create data loaders for train and validation sets
    dataset.set_split(is_train=True)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    dataset.set_split(is_train=False)
    val_loader = DataLoader(dataset, batch_size=batch_size)
    
    # Initialize lists to store losses
    train_losses = []
    val_losses = []
    
    # Print dataset information
    info = dataset.get_dataset_info()
    print("\nDataset Information:")
    print(f"Total samples: {info['total_samples']}")
    print(f"Used samples: {info['used_samples']}")
    print(f"Training samples: {info['train_samples']}")
    print(f"Validation samples: {info['val_samples']}")
    print(f"Vocabulary size: {info['vocab_size']}")
    print(f"Max sequence length: {info['max_len']}\n")
    
    for epoch in range(epochs):
        # Training
        dataset.set_split(is_train=True)
        model.train()
        total_train_loss = 0
        train_progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        
        for batch_idx, (src, tgt) in enumerate(train_progress):
            src, tgt = src.to(device), tgt.to(device)
            
            optimizer.zero_grad()
            output = model(src)
            
            loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_train_loss += loss.item()
            
            # Update progress bar
            train_progress.set_postfix({'loss': loss.item()})
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        dataset.set_split(is_train=False)
        model.eval()
        total_val_loss = 0
        val_progress = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
        
        with torch.no_grad():
            for src, tgt in val_progress:
                src, tgt = src.to(device), tgt.to(device)
                output = model(src)
                loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))
                total_val_loss += loss.item()
                
                # Update progress bar
                val_progress.set_postfix({'loss': loss.item()})
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}')
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
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
    
    # Load dataset with sampling
    dataset = KyrgyzTextDataset(
        'example_dataset.tsv',
        max_len=512,
        sample_ratio=1,     # Use 25% of the data
        val_ratio=0.1,      # 10% of that 25% will be validation
        seed=42             # For reproducibility
    )
    
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
    train_losses, val_losses = train(
        model=model,
        dataset=dataset,
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
