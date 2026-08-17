# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Malay "rojak" (informal/colloquial) text lemmatizer built by fine-tuning Llama 3.2 3B with QLoRA using the Unsloth framework. The system learns to convert informal Malay text containing slang, abbreviations, code-switching (Malay-English), and non-standard spellings into their standard lemmatized forms.

**Key challenge**: The lemmatizer must handle ambiguous tokens where the dictionary returns multiple candidates, out-of-vocabulary abbreviations, elongated words, reduplication patterns, and code-switched English.

## Three-Stage Pipeline

The entire workflow is controlled by `llm.py` with three distinct stages that MUST be run sequentially:

### Stage 1: Data Preparation
```bash
python llm.py --prepare
```
- Reads `seed.jsonl` (created by `databuilder.py`)
- Generates train/validation splits in ChatML format
- Outputs: `train_chatml.jsonl`, `val_chatml.jsonl`, `val_seed.jsonl`
- Reports corpus statistics by token difficulty category (dict/slang/oov/ambiguous/etc.)

### Stage 2: Training
```bash
python llm.py --train
```
- Fine-tunes `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` with QLoRA
- Uses 4-bit quantization (required for 8GB VRAM RTX 4060)
- Saves LoRA adapter to `lemmatizer_lora/` directory
- Training output stored in `training_runs/`

**Critical**: The SYSTEM_PROMPT defined in llm.py:42-45 must be BYTE-IDENTICAL between training and inference. Any change breaks the model.

### Stage 3: Evaluation
```bash
python llm.py --eval
```
- Loads the fine-tuned adapter from `lemmatizer_lora/`
- Compares model predictions against rule-based baseline (first dictionary candidate)
- Reports accuracy by token source category (dict/slang/oov/ambiguous/etc.)
- Shows format failures and overall delta improvement

## Data Pipeline Architecture

1. **databuilder.py**: Creates the seed dataset
   - Contains hardcoded lexicon (LEX dictionary) mapping tokens to: (lemma, language, source_category, dictionary_candidates)
   - Defines 50+ sample sentences covering all rojak patterns
   - Outputs `seed.jsonl` with full annotation including candidates for baseline evaluation
   - Run: `python databuilder.py`

2. **llm.py --prepare**: Transforms seed into training format
   - Converts to ChatML format with system/user/assistant roles
   - Splits into train/val preserving val_seed.jsonl for evaluation

3. **llm.py --train**: QLoRA fine-tuning
   - LoRA rank=16, targets all attention + MLP projections
   - Gradient checkpointing via Unsloth (lower VRAM than standard)
   - Loss computed ONLY on assistant response (prompt is masked)
   - Max sequence length=512 (rojak sentences are short)

4. **llm.py --eval**: Measure improvement over baseline
   - Baseline = first dictionary candidate, else token itself
   - Model must generate exact format: "token -> lemma" per line
   - Format failures counted separately (model output unparseable)

## Key Configuration Parameters

Located in llm.py:25-45:
- `BASE_MODEL`: unsloth/Llama-3.2-3B-Instruct-bnb-4bit
- `MAX_SEQ_LEN`: 512 (main VRAM control lever)
- `ADAPTER_DIR`: lemmatizer_lora
- `VAL_RATIO`: 0.15
- `SYSTEM_PROMPT`: Critical - do not modify without retraining

Training hyperparameters (llm.py:162-179):
- Batch size: 2 (reduce to 1 if OOM, raise gradient_accumulation_steps to 8)
- Gradient accumulation: 4
- Epochs: 3
- Learning rate: 2e-4
- Optimizer: adamw_8bit

## Token Difficulty Categories

The lexicon in databuilder.py categorizes each token by its source type:
- **dict**: Dictionary resolves cleanly to single lemma
- **slang**: Informal but dictionary returns candidates (e.g., "dah" -> "sudah")
- **oov**: Out-of-vocabulary abbreviations (e.g., "xleh" -> "tidak boleh")
- **ambiguous**: Dictionary returns multiple candidates, context needed
- **particle**: Discourse particles kept as-is (la, weh, kan)
- **elongation**: Repeated letters for emphasis (gilaa -> gila)
- **reduplication**: Number-2 pattern (budak2 -> budak)
- **codeswitch**: English words in Malay sentences
- **time/date/money**: Numeric normalization

The fine-tune is being asked to fix everything EXCEPT "dict" category tokens.

## GPU and Environment

- Designed for 8GB VRAM (RTX 4060)
- Requires WSL2 or Linux (native Windows not recommended per llm.py:12)
- Dependencies: unsloth, trl, peft, accelerate, bitsandbytes
- Uses bf16 on Ada architecture, fp16 fallback for older GPUs
- `gpu.py` available for CUDA verification

## Output Format Contract

The model MUST generate this exact format:
```
token -> lemma
token -> lemma
...
```

Parsed by LINE_RE regex in llm.py:204. Any deviation counts as format failure. Multi-word lemmas allowed (e.g., "xleh" -> "tidak boleh").

## Files and Directories

- `llm.py`: Main training script (prepare/train/eval stages)
- `databuilder.py`: Creates seed.jsonl from hardcoded lexicon
- `gpu.py`: CUDA diagnostic utility
- `seed.jsonl`: Full annotated dataset (created by databuilder.py)
- `train_chatml.jsonl`, `val_chatml.jsonl`: ChatML formatted splits
- `val_seed.jsonl`: Validation set with original annotations for eval
- `lemmatizer_lora/`: Saved LoRA adapter + tokenizer
- `training_runs/`: Checkpoints and training logs
- `unsloth_compiled_cache/`: Unsloth-specific trainer implementations

## Deployment Notes

For llama.cpp/Ollama deployment, uncomment llm.py:197 to export GGUF:
```python
model.save_pretrained_gguf("lemmatizer_gguf", tokenizer, quantization_method="q4_k_m")
```

## Common Modifications

**To add new vocabulary**:
1. Update LEX dictionary in databuilder.py with new tokens
2. Add example sentences to SENTENCES list
3. Rerun: `python databuilder.py` -> `python llm.py --prepare` -> `python llm.py --train`

**To tune for lower VRAM**:
1. Reduce MAX_SEQ_LEN in llm.py (512 -> 256)
2. Reduce batch size to 1, increase gradient_accumulation_steps to 8

**To change base model**:
1. Update BASE_MODEL and CHAT_TEMPLATE in llm.py
2. Verify special tokens in training output (llm.py:154-156) - must see actual tokens, not escaped strings