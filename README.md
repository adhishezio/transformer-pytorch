# Transformer From Scratch

This is a simple implementation of the Transformer model based on the paper [Attention is All You Need](https://arxiv.org/abs/1706.03762). I built it from scratch in PyTorch to better understand how each part of the architecture works.

## What this project does

- Implements the full Transformer architecture (no `torch.nn.Transformer` used)
- Covers encoder and decoder blocks with attention and feedforward layers
- Adds positional encodings and residual connections
- Uses Hugging Face `datasets` and `tokenizers` to handle real bilingual translation data
- Trains the model with a custom loop and tracks performance using BLEU, WER, and CER


## How to run it

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train the model:

```bash
python train.py
```

## Based on 
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [Attention is All You Need](https://arxiv.org/abs/1706.03762)

This project helped me learn how Transformers work at a deeper level. Feel free to try it out or build on top of it!