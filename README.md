# Kyrgyz diacritics restorer

The project uses PyTorch to restore `ө`, `ү`, and `ң` in Kyrgyz text typed with `о`, `у`, and `н`. Dependencies are in `requirements.txt`.

Expected result:

```text
Input:     мен уйронуп жатам
Expected:  мен үйрөнүп жатам
```

Note: this shows the correct spelling. We haven't tested the model's output for this example.

## How it works

The model reads text one character at a time using a Transformer. It processes the whole input in one pass, then chooses between `о/ө`, `у/ү`, or `н/ң` at each position. Other characters in its vocabulary stay unchanged after lowercasing.

Each position can use the current character and the text before it. It cannot use the text after it. The model does not read its own corrections.

Training adds `<BOS>` at the start and `<EOS>` at the end. Input and target characters have matching positions.

## Run the model

`inference_kaggle.ipynb` -- loads the saved model and restores text. Open it in Kaggle, turn on internet access, and run the cells in order. It uses a GPU if available, otherwise a CPU.

The notebook downloads `model.pt`, `vocab.json`, and `config.json` from [murat/ky-diacritics-restorer on Hugging Face](https://huggingface.co/murat/ky-diacritics-restorer). These contain the saved weights, character IDs, and model settings.

Note: model weights are not included in this Git repository.

After running the notebook's setup cells:

```python
pipeline = DiacriticsRestorerPipeline("murat/ky-diacritics-restorer")

print(pipeline.restore("мен уйронуп жатам"))

results = pipeline.restore_batch([
    "кыргызстан онугуп кетет",
    "биз омур бою окууга даярбыз",
])
```

`restore_batch` processes one text at a time.

## Setup for local training

```bash
git clone https://github.com/jumasheff/ky_diacritics_restorer.git
cd ky_diacritics_restorer

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install matplotlib
```

## Training files

`train.py` -- trains locally and logs results to Weights & Biases. Runs for 10 epochs by default, with batch size 32 and learning rate `1e-4`. Saves weights every 5 epochs to `model_epoch_<epoch>.pt`.

`train_kaggle.ipynb` -- trains on Kaggle and uploads the model to Hugging Face. Currently uses `example_dataset.tsv` for a demo run.

`train_kaggle_ddp.ipynb` -- experimental training with multiple GPUs.

`train_colab.ipynb` -- training notebook for Colab.

Before running a notebook, check the dataset path, how much data it uses, and which Hugging Face repo it uploads to.

Note: `train.py` saves weights only. Keep the matching character IDs and model settings too. You need them to load the model later.

## Prepare the data

`create_dataset.py` -- reads `all_texts.txt` and writes pairs to `dataset.tsv`. It lowercases the text and replaces `ө` with `о`, `ү` with `у`, and `ң` with `н`. It skips lines where nothing changed.

The TSV file has three columns:

```text
id	source	target
1	мен уйронуп жатам	мен үйрөнүп жатам
```

Note: the loader in `model.py` currently reads these columns backwards. It uses `target` as input and `source` as the expected output. This affects the generated data and `example_dataset.tsv`. Fix the column order in the loader before retraining.

## Current limits

- All output is lowercase.
- Unknown characters become `<UNK>`.
- With `max_len=512`, input can have up to 510 characters. The code does not split or shorten longer text.
- The model cannot look ahead when choosing a letter.
- We haven't checked accuracy after the `<BOS>` position fix.

## Tests

Install PyTorch and NumPy, then run this from the project folder:

```bash
python -m unittest discover -s tests -v
```

The test checks `model.py` and the inference notebook with fixed scores in place of a trained model. It checks character positions after `<BOS>`, allowed letter choices, unchanged characters, and empty input.

It does not test the saved model's accuracy.
