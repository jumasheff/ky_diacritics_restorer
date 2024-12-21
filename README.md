# Kyrgyz Diacritics Restorer

This project implements a sequence-to-sequence model to restore diacritics in Kyrgyz text. It can convert text without diacritics (о, у, н) to text with proper diacritics (ө, ү, ң).

## Setup

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Prepare your training data:
The `example.txt` file contains pairs of correct and diacritic-less text, separated by '|'. Each line should be in the format:
```
Correct text with diacritics | Text without diacritics
```

## Training

To train the model, run:
```bash
python train.py
```

The script will:
- Train the model for 100 epochs
- Save the best model based on loss
- Print training progress
- Test the model on example sentences every 10 epochs

## Model Architecture

The model uses:
- Encoder-Decoder architecture with LSTM
- Character-level embeddings
- Teacher forcing during training
- Attention mechanism for better sequence alignment

## Hyperparameters

The current hyperparameters are:
- Embedding dimension: 256
- Hidden dimension: 512
- Number of layers: 2
- Dropout: 0.2
- Batch size: 32
- Learning rate: 0.001

You can modify these in `train.py` to experiment with different configurations.
