import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from model import KyrgyzTextDataset, TransformerDiacriticsRestorer, train_model, restore_diacritics
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def validate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            src = batch['input'].to(device)
            trg = batch['target'].to(device)
            padding_mask = src == model.embedding.num_embeddings - 1
            
            output = model(src, src_padding_mask=padding_mask)
            
            output_dim = output.shape[-1]
            output = output.reshape(-1, output_dim)
            trg = trg.reshape(-1)
            
            loss = criterion(output, trg)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def main():
    # Set random seed for reproducibility
    set_seed(42)
    
    # Hyperparameters
    D_MODEL = 256  # Embedding & transformer dimension
    N_HEAD = 8     # Number of attention heads
    N_LAYERS = 6   # Number of transformer layers
    DIM_FEEDFORWARD = 1024  # Feedforward dimension
    DROPOUT = 0.1
    BATCH_SIZE = 64
    N_EPOCHS = 100
    LEARNING_RATE = 0.0003
    PATIENCE = 5   # Early stopping patience
    VAL_SPLIT = 0.1  # 10% for validation
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    dataset = KyrgyzTextDataset('dataset.tsv')
    
    # Split into train and validation sets
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    
    # Initialize model
    model = TransformerDiacriticsRestorer(
        vocab_size=dataset.vocab_size,
        d_model=D_MODEL,
        nhead=N_HEAD,
        num_layers=N_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT
    ).to(device)
    
    # Loss, optimizer and scheduler
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.pad_idx)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(N_EPOCHS):
        # Training
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        
        # Validation
        val_loss = validate_model(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f'Epoch: {epoch+1}')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        
        # Save best model and early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, 'best_model.pt')
            print(f'Model saved with validation loss: {best_val_loss:.4f}')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print('Early stopping triggered')
                break
        
        # Test the model on examples every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()
            test_sentences = [
                "менин уйум чон.",
                "кочодо коп машина бар.",
                "бугун кун ысык.",
                "ал китеп окуганды жакшы корот.",
                "биз тоого чыгып, таза аба жутабыз."
            ]
            print("\nTesting examples:")
            for sent in test_sentences:
                restored = restore_diacritics(model, sent, dataset, device)
                print(f'Input:  {sent}')
                print(f'Output: {restored}\n')

if __name__ == "__main__":
    main()
