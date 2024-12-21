import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from typing import List, Dict, Tuple, Optional
import math
import random

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return x + self.pe[:x.size(0)]

class KyrgyzTextDataset(Dataset):
    def __init__(self, file_path: str, max_len: int = 512, sample_ratio: float = 1.0, val_ratio: float = 0.1, seed: int = 42):
        """
        Initialize the dataset.
        
        Args:
            file_path (str): Path to the TSV file
            max_len (int): Maximum sequence length
            sample_ratio (float): Ratio of data to use (0.0 to 1.0)
            val_ratio (float): Ratio of data to use for validation
            seed (int): Random seed for reproducibility
        """
        self.input_texts = []
        self.target_texts = []
        self.max_len = max_len
        
        # Set random seed for reproducibility
        random.seed(seed)
        
        # Special tokens
        self.special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        self.char_to_idx = {token: idx for idx, token in enumerate(self.special_tokens)}
        self.idx_to_char = {idx: token for idx, token in enumerate(self.special_tokens)}
        
        self.pad_idx = self.char_to_idx['<PAD>']
        self.unk_idx = self.char_to_idx['<UNK>']
        self.bos_idx = self.char_to_idx['<BOS>']
        self.eos_idx = self.char_to_idx['<EOS>']
        
        # Read all data first
        all_input_texts = []
        all_target_texts = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            next(f)  # Skip header
            lines = f.readlines()
            
            # Shuffle lines
            random.shuffle(lines)
            
            # Calculate how many lines to use based on sample_ratio
            num_lines = int(len(lines) * sample_ratio)
            lines = lines[:num_lines]
            
            for line in lines:
                if '\t' in line:
                    fields = line.strip().split('\t')
                    if len(fields) >= 2:
                        target = fields[1].strip().lower()
                        input_text = fields[2].strip().lower() if len(fields) > 2 else target
                        
                        if len(input_text) <= max_len - 2:  # -2 for BOS and EOS
                            all_input_texts.append(input_text)
                            all_target_texts.append(target)
                            
                            # Update vocabulary
                            for char in input_text + target:
                                if char not in self.char_to_idx:
                                    idx = len(self.char_to_idx)
                                    self.char_to_idx[char] = idx
                                    self.idx_to_char[idx] = char
        
        self.vocab_size = len(self.char_to_idx)
        
        # Store dataset statistics
        self.total_samples = len(all_input_texts)
        self.used_samples = int(self.total_samples * sample_ratio)
        
        # Split into train and validation sets
        val_size = int(len(all_input_texts) * val_ratio)
        train_size = len(all_input_texts) - val_size
        
        self.train_indices = list(range(train_size))
        self.val_indices = list(range(train_size, len(all_input_texts)))
        
        self.input_texts = all_input_texts
        self.target_texts = all_target_texts
        
        self.is_train = True  # Flag to switch between train and validation sets
    
    def set_split(self, is_train: bool = True):
        """Switch between train and validation sets"""
        self.is_train = is_train
    
    def __len__(self) -> int:
        if self.is_train:
            return len(self.train_indices)
        return len(self.val_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Map idx to the correct index based on split
        if self.is_train:
            actual_idx = self.train_indices[idx]
        else:
            actual_idx = self.val_indices[idx]
        
        input_text = self.input_texts[actual_idx]
        target_text = self.target_texts[actual_idx]
        
        # Convert to indices with BOS and EOS tokens
        input_indices = [self.bos_idx] + [self.char_to_idx.get(c, self.unk_idx) for c in input_text] + [self.eos_idx]
        target_indices = [self.bos_idx] + [self.char_to_idx.get(c, self.unk_idx) for c in target_text] + [self.eos_idx]
        
        # Pad sequences
        input_indices = input_indices + [self.pad_idx] * (self.max_len - len(input_indices))
        target_indices = target_indices + [self.pad_idx] * (self.max_len - len(target_indices))
        
        return (torch.tensor(input_indices[:self.max_len], dtype=torch.long),
                torch.tensor(target_indices[:self.max_len], dtype=torch.long))
    
    def get_splits(self):
        """Return two dataset objects for train and validation"""
        train_dataset = KyrgyzTextDataset(self.file_path, self.max_len)
        train_dataset.set_split(True)
        
        val_dataset = KyrgyzTextDataset(self.file_path, self.max_len)
        val_dataset.set_split(False)
        
        return train_dataset, val_dataset
    
    def get_dataset_info(self) -> Dict:
        """Return information about the dataset"""
        return {
            'total_samples': self.total_samples,
            'used_samples': self.used_samples,
            'train_samples': len(self.train_indices),
            'val_samples': len(self.val_indices),
            'vocab_size': self.vocab_size,
            'max_len': self.max_len
        }

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        src2 = self.self_attn(src, src, src, attn_mask=src_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class DiacriticsRestorer(nn.Module):
    def __init__(self, 
                 vocab_size: int,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 max_len: int = 512):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        encoder_layers = [
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ]
        self.encoder_layers = nn.ModuleList(encoder_layers)
        
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # src shape: [batch_size, seq_len]
        
        # Create attention mask
        src_mask = self.generate_square_subsequent_mask(src.size(1)).to(src.device)
        
        # Embedding and positional encoding
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(x)
        
        # Encoder layers
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        
        # Output projection
        output = self.output_projection(x)
        return output

def train_epoch(model: nn.Module,
                dataloader: DataLoader,
                optimizer: torch.optim.Optimizer,
                criterion: nn.Module,
                device: torch.device) -> float:
    model.train()
    total_loss = 0
    
    for batch_idx, (src, tgt) in enumerate(dataloader):
        src, tgt = src.to(device), tgt.to(device)
        
        optimizer.zero_grad()
        output = model(src)
        
        loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def validate(model: nn.Module,
            dataloader: DataLoader,
            criterion: nn.Module,
            device: torch.device) -> float:
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            output = model(src)
            loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

@torch.no_grad()
def restore_diacritics(model: nn.Module,
                      text: str,
                      dataset: KyrgyzTextDataset,
                      device: torch.device) -> str:
    model.eval()
    
    # Prepare input
    input_indices = [dataset.bos_idx] + [dataset.char_to_idx.get(c, dataset.unk_idx) for c in text.lower()] + [dataset.eos_idx]
    input_tensor = torch.tensor(input_indices).unsqueeze(0).to(device)  # Add batch dimension
    
    # Generate output
    output = model(input_tensor)
    
    # Restrict predictions based on input character
    result = []
    for i, idx in enumerate(input_indices[1:-1]):  # Skip BOS and EOS
        char = dataset.idx_to_char[idx]
        
        if char in ['о', 'у', 'н']:
            if char == 'о':
                candidates = [dataset.char_to_idx['о'], dataset.char_to_idx['ө']]
            elif char == 'у':
                candidates = [dataset.char_to_idx['у'], dataset.char_to_idx['ү']]
            elif char == 'н':
                candidates = [dataset.char_to_idx['н'], dataset.char_to_idx['ң']]
            
            # Mask all logits except candidates
            mask = torch.full_like(output[:, i, :], float('-inf'))
            for c in candidates:
                mask[:, c] = 0
            output[:, i, :] += mask
            
            pred_idx = output[:, i, :].argmax(dim=-1).item()
            result.append(dataset.idx_to_char[pred_idx])
        else:
            # For non-ambiguous characters, use the original character
            result.append(char)
    
    return ''.join(result)
