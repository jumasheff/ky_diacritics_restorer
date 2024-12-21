import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class KyrgyzTextDataset(Dataset):
    def __init__(self, file_path):
        self.input_texts = []
        self.target_texts = []
        
        # Create character to index mappings
        self.char_to_idx = {}
        self.idx_to_char = {}
        
        # Read data and build vocabulary
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if '|' in line:
                    target, input_text = line.strip().split('|')
                    target = target.strip()
                    input_text = input_text.strip()
                    self.input_texts.append(input_text)
                    self.target_texts.append(target)
                    
                    # Update vocabulary
                    for char in input_text + target:
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

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim)  # +1 for padding
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return outputs, hidden, cell

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim)  # +1 for padding
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size + 1)  # +1 for padding
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc_out.out_features
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # Encoder
        encoder_outputs, hidden, cell = self.encoder(src)
        
        # First decoder input is first target token
        decoder_input = trg[:, 0]
        
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[:, t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            decoder_input = trg[:, t] if teacher_force else top1
            
        return outputs

def train_model(model, train_loader, optimizer, criterion, device, clip=1):
    model.train()
    epoch_loss = 0
    
    for batch in train_loader:
        src = batch['input'].to(device)
        trg = batch['target'].to(device)
        
        optimizer.zero_grad()
        output = model(src, trg)
        
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        
        loss = criterion(output, trg)
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
        
        # Encoder
        encoder_outputs, hidden, cell = model.encoder(src)
        
        # Initialize decoder input
        decoder_input = src[:, 0]
        
        output_chars = []
        max_length = len(text) + 50  # Prevent infinite loop
        
        for _ in range(max_length):
            output, hidden, cell = model.decoder(decoder_input, hidden, cell)
            top1 = output.argmax(1)
            output_chars.append(dataset.idx_to_char[top1.item()])
            decoder_input = top1
            
            if len(output_chars) >= len(text):
                break
                
    return ''.join(output_chars)
