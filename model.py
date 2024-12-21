import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class KyrgyzTextDataset(Dataset):
    def __init__(self, file_path):
        self.input_texts = []
        self.target_texts = []
        
        # Create character to index mappings
        self.char_to_idx = {}
        self.idx_to_char = {}
        
        # Read data and build vocabulary
        with open(file_path, 'r', encoding='utf-8') as f:
            next(f)  # Skip header
            for line in f:
                id_, source, target = line.strip().split('\t')
                self.input_texts.append(source)
                self.target_texts.append(target)
                
                # Update vocabulary
                for char in source + target:
                    if char not in self.char_to_idx:
                        idx = len(self.char_to_idx)
                        self.char_to_idx[char] = idx
                        self.idx_to_char[idx] = char
        
        self.vocab_size = len(self.char_to_idx)
        self.pad_idx = self.vocab_size
        self.char_to_idx['<PAD>'] = self.pad_idx
        self.idx_to_char[self.pad_idx] = '<PAD>'
        
    def __len__(self):
        return len(self.input_texts)
    
    def __getitem__(self, idx):
        input_text = self.input_texts[idx]
        target_text = self.target_texts[idx]
        
        # Convert to indices
        input_indices = [self.char_to_idx[c] for c in input_text]
        target_indices = [self.char_to_idx[c] for c in target_text]
        
        return {
            'input': torch.tensor(input_indices, dtype=torch.long),
            'target': torch.tensor(target_indices, dtype=torch.long),
            'input_text': input_text,
            'target_text': target_text
        }

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerDiacriticsRestorer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Character embedding
        self.embedding = nn.Embedding(vocab_size + 1, d_model)  # +1 for padding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer
        self.output_layer = nn.Linear(d_model, vocab_size + 1)  # +1 for padding
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask=None, src_padding_mask=None):
        # Embed characters and add positional encoding
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        src = self.dropout(src)
        
        # Transform through encoder
        output = self.transformer_encoder(src, src_mask, src_padding_mask)
        
        # Project to vocabulary size
        output = self.output_layer(output)
        return output

def create_padding_mask(seq, pad_idx):
    return seq == pad_idx

def train_model(model, train_loader, optimizer, criterion, device, clip=1):
    model.train()
    epoch_loss = 0
    
    for batch in train_loader:
        src = batch['input'].to(device)
        trg = batch['target'].to(device)
        
        # Create padding mask
        padding_mask = create_padding_mask(src, model.embedding.num_embeddings - 1)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(src, src_padding_mask=padding_mask)
        
        # Reshape output and target for loss calculation
        output_dim = output.shape[-1]
        output = output.reshape(-1, output_dim)
        trg = trg.reshape(-1)
        
        # Calculate loss
        loss = criterion(output, trg)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(train_loader)

def restore_diacritics(model, text, dataset, device):
    model.eval()
    with torch.no_grad():
        # Convert input text to indices
        input_indices = [dataset.char_to_idx[c] for c in text]
        src = torch.tensor(input_indices).unsqueeze(0).to(device)
        
        # Forward pass through the model
        output = model(src)
        
        # Get predictions
        predictions = output.argmax(dim=-1)
        
        # Convert predictions to characters
        output_chars = [dataset.idx_to_char[idx.item()] for idx in predictions[0]]
        
    return ''.join(output_chars)
