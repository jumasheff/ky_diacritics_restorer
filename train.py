import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import KyrgyzTextDataset, Encoder, Decoder, Seq2Seq, train_model, restore_diacritics
from tqdm import tqdm

def main():
    # Hyperparameters
    EMBEDDING_DIM = 256
    HIDDEN_DIM = 512
    N_LAYERS = 2
    DROPOUT = 0.2
    BATCH_SIZE = 32
    N_EPOCHS = 100
    LEARNING_RATE = 0.001
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    dataset = KyrgyzTextDataset('example.txt')
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize model
    encoder = Encoder(dataset.vocab_size + 1, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT)
    decoder = Decoder(dataset.vocab_size + 1, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT)
    model = Seq2Seq(encoder, decoder, device).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(N_EPOCHS):
        loss = train_model(model, train_loader, optimizer, criterion, device)
        print(f'Epoch: {epoch+1}, Loss: {loss:.4f}')
        
        # Save best model
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), 'best_model.pt')
            print(f'Model saved with loss: {best_loss:.4f}')
        
        # Test the model on a few examples
        if (epoch + 1) % 10 == 0:
            model.eval()
            test_sentences = [
                "Менин уйум чон.",
                "Кочодо коп машина бар.",
                "Бугун кун ысык."
            ]
            print("\nTesting examples:")
            for sent in test_sentences:
                restored = restore_diacritics(model, sent, dataset, device)
                print(f'Input: {sent}')
                print(f'Output: {restored}\n')

if __name__ == "__main__":
    main()
