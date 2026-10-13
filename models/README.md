# models

Fine-tuned LoRA adapters go here, one folder each:

```
models/lemmatizer_lora_llama/
models/lemmatizer_lora_qwen/
```

`python train.py --model llama` (or `qwen`) writes them here. If you trained
an adapter elsewhere, copy its whole folder in - the app and `main.py` pick it
up automatically.

Adapters are not stored in git: each one is over 100 MB.
