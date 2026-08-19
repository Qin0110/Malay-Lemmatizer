#!/usr/bin/env python
"""
Malay Rojak Lemmatizer - Llama 3.2 3B QLoRA fine-tune with Unsloth.
Target format: LEMMA SENTENCE (not token -> lemma pairs).

    python train_lemmatizer.py --prepare
    python train_lemmatizer.py --train
    python train_lemmatizer.py --eval

Install:
    pip install triton-windows      # native Windows only
    pip install unsloth
"""
import os
import argparse
import json
import random
import re
from collections import defaultdict
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------

SEED_FILE = "seed500.jsonl"
TRAIN_FILE = "data/training.jsonl"     # NOT train_chatml.jsonl - build_500.py owns that
VAL_FILE = "data/val_sft.jsonl"
VAL_SEED_FILE = "data/val_seed.jsonl"
ADAPTER_DIR = "lemmatizer_lora"

BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
MAX_SEQ_LEN = 512
CHAT_TEMPLATE = "llama-3.2"

VAL_RATIO = 0.15
SEED = 3407
ON_WINDOWS = True                       # set False under WSL2/Linux

# Must match build_500.py exactly, and be byte-identical at inference.
SYSTEM_PROMPT = (
    "Anda adalah pelemma bahasa Melayu. Tukarkan ayat rojak kepada ayat "
    "berbentuk lema. Kembalikan ayat sahaja."
)


# ----------------------------------------------------------------------------
# STAGE 1: PREPARE
# ----------------------------------------------------------------------------

def to_chatml(row):
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": row["text"]},
        {"role": "assistant", "content": row["lemma_text"]},
    ]}


def prepare():
    os.makedirs("data", exist_ok=True)

    rows = [json.loads(l) for l in open(SEED_FILE, encoding="utf-8") if l.strip()]
    missing = [r["id"] for r in rows if "lemma_text" not in r]
    if missing:
        raise SystemExit(f"{len(missing)} rows lack lemma_text, e.g. {missing[:3]}")

    random.Random(SEED).shuffle(rows)
    n_val = max(1, int(len(rows) * VAL_RATIO))
    val_rows, train_rows = rows[:n_val], rows[n_val:]

    def dump(path, data, fn):
        with open(path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(fn(r), ensure_ascii=False) + "\n")

    dump(TRAIN_FILE, train_rows, to_chatml)
    dump(VAL_FILE, val_rows, to_chatml)
    dump(VAL_SEED_FILE, val_rows, lambda r: r)

    counts, verified = defaultdict(int), 0
    review = {"affix", "affix_guess", "unknown", "codeswitch_guess", "ms_unlisted"}
    for r in rows:
        for e in r["lemmas"]:
            counts[e["src"]] += 1
            if e["src"] not in review:
                verified += 1

    total = sum(counts.values())
    print(f"Train {len(train_rows)} / Val {len(val_rows)} rows, {total} tokens\n")
    for src, n in sorted(counts.items(), key=lambda x: -x[1]):
        flag = "  <-- silver" if src in review else ""
        print(f"  {src:<18} {n:>5}  ({n / total:>5.1%}){flag}")
    print(f"\n  Labels you can trust: {verified / total:.1%}")
    print(f"  Longest input:  {max(len(r['text'].split()) for r in rows)} words")
    print(f"  Longest target: {max(len(r['lemma_text'].split()) for r in rows)} words")


# ----------------------------------------------------------------------------
# STAGE 2: TRAIN
# ----------------------------------------------------------------------------

def train():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    tokenizer = get_chat_template(tokenizer, chat_template=CHAT_TEMPLATE)

    def fmt(examples):
        return {"text": [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in examples["messages"]
        ]}

    train_ds = load_dataset("json", data_files=TRAIN_FILE, split="train").map(fmt, batched=True)

    print("\n" + "=" * 70)
    print(train_ds[0]["text"])
    print("=" * 70 + "\n")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,              # TRL >=0.12: rename to processing_class=
        train_dataset=train_ds,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LEN,
            dataset_num_proc=1 if ON_WINDOWS else 2,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=3,
            learning_rate=2e-4,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=5,
            seed=SEED,
            output_dir="training_runs",
            report_to="none",
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    stats = trainer.train()
    print(f"\nRuntime: {stats.metrics['train_runtime'] / 60:.1f} min")

    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(f"Adapter saved to {ADAPTER_DIR}/")


# ----------------------------------------------------------------------------
# STAGE 3: EVALUATE
# ----------------------------------------------------------------------------

def norm(w):
    return re.sub(r"[^\w]", "", w.lower())


def align(pred_words, lemmas):
    """Map predicted words back to source tokens.

    A gold lemma may be multi-word (`xleh` -> `tidak boleh`), so positions do
    not line up 1:1. We consume len(gold.split()) words per token. If the total
    count disagrees, alignment is unsafe and we report it instead of guessing.
    """
    expected = sum(len(e["l"].split()) for e in lemmas)
    if len(pred_words) != expected:
        return None
    out, i = [], 0
    for e in lemmas:
        k = len(e["l"].split())
        out.append(" ".join(pred_words[i:i + k]))
        i += k
    return out


def baseline_predict(row):
    return [(row["candidates"].get(e["t"]) or [e["t"]])[0] for e in row["lemmas"]]


def evaluate():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ADAPTER_DIR, max_seq_length=MAX_SEQ_LEN,
        dtype=None, load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template=CHAT_TEMPLATE)
    FastLanguageModel.for_inference(model)

    rows = [json.loads(l) for l in open(VAL_SEED_FILE, encoding="utf-8") if l.strip()]
    hits = defaultdict(lambda: {"model": 0, "base": 0, "n": 0})
    exact, misaligned = 0, 0

    for row in rows:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["text"]}]
        ids = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
        out = model.generate(input_ids=ids, max_new_tokens=160, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
        pred = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

        # Metric 1: whole-sentence exact match
        if norm(pred.replace(" ", "")) == norm(row["lemma_text"].replace(" ", "")):
            exact += 1

        # Metric 2: per-token, by difficulty category
        aligned = align(pred.split(), row["lemmas"])
        if aligned is None:
            misaligned += 1
        base = baseline_predict(row)
        for i, e in enumerate(row["lemmas"]):
            d = hits[e["src"]]
            d["n"] += 1
            if aligned and norm(aligned[i]) == norm(e["l"]):
                d["model"] += 1
            if norm(base[i]) == norm(e["l"]):
                d["base"] += 1

    print(f"\nSentence exact match: {exact}/{len(rows)} ({exact / len(rows):.1%})")
    print(f"Length misalignment:  {misaligned}/{len(rows)} ({misaligned / len(rows):.1%})\n")
    print(f"{'category':<18} {'n':>5} {'baseline':>10} {'finetune':>10} {'delta':>8}")
    print("-" * 55)
    tot = {"model": 0, "base": 0, "n": 0}
    for src, d in sorted(hits.items(), key=lambda x: -x[1]["n"]):
        b, m = d["base"] / d["n"], d["model"] / d["n"]
        print(f"{src:<18} {d['n']:>5} {b:>9.1%} {m:>9.1%} {m - b:>+8.1%}")
        for k in tot:
            tot[k] += d[k]
    b, m = tot["base"] / tot["n"], tot["model"] / tot["n"]
    print("-" * 55)
    print(f"{'OVERALL':<18} {tot['n']:>5} {b:>9.1%} {m:>9.1%} {m - b:>+8.1%}")
    print("\nNote: `baseline` here is build_500.py's own rule output, so it scores")
    print("~100% on dict/slang by construction. Swap in MALINDO Morph for a real one.")


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--train", action="store_true")
    p.add_argument("--eval", action="store_true")
    a, _ = p.parse_known_args()          # tolerate Jupyter's -f kernel.json

    if a.prepare:
        prepare()
    elif a.train:
        train()
    elif a.eval:
        evaluate()
    else:
        p.print_help()