# Iludev

A small GPT-style chat model, trained from scratch in PyTorch by SiLuyla Development.

| File | What it does |
|---|---|
| `train.py` | data prep, pretraining + chat fine-tuning, eval, release folder |
| `chat.py` | standalone CLI chat |
| `export_gguf.py` | converts a trained model to GGUF for llama.cpp / llama-cpp-python / Ollama |
| `iludev.ipynb` | Colab recipe |

## Quick start

```
pip install torch tokenizers datasets safetensors
python train.py prep  --out out --size m --pre-mb 4500
python train.py train --out out --size m --hours 12
python train.py eval  --out out
python train.py chat  --out out
python chat.py chat --model out/chat.safetensors
python export_gguf.py --model out/chat.safetensors --out iludev.gguf --verify
python train.py release --out out --release-name "Xeriosa 1"
```
