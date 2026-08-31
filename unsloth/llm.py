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
import time
import argparse
import json
import random
import re
from collections import defaultdict

import skill
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

BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
CHAT_TEMPLATE = "llama-3.2"

VAL_RATIO = 0.15
SEED = 3407
ON_WINDOWS = True                       # set False under WSL2/Linux

# --- training budget --------------------------------------------------------
# Three things grew 2.7x when the corpus went from 3,000 to 8,000 sentences:
# the number of optimiser steps, the size of every evaluation, and the number
# of checkpoint writes. Evaluation grew worst - 800 dev rows scored every 50
# steps is 7.2x the forward passes the 300-row dev set cost - and none of it
# changes the trained weights, it only picks the checkpoint.
#
# EPOCHS is 2, not 3, on evidence rather than impatience: both previous runs
# chose their best checkpoint before the end of epoch 2 (step 700 of 1,014,
# and step 650 of 1,012). The third epoch overfitted both times.
EPOCHS = 2
BUDGET_MINS = 120        # hard wall-clock cap; 0 disables it
BATCH_SIZE = 8           # sequences average 79 tokens, so batch 2 left the
ACCUM_STEPS = 1          # GPU waiting on kernel launches. 8x1 == 2x4.
EVAL_ROWS = 200          # dev rows used *in the loop* to pick the checkpoint
EVAL_EVERY = 100         # steps between evaluations (and checkpoint writes)

# v2 trains on every corpus the registry links (sources.py) and teaches the
# model to show its derivation (skill.py). v1 trained on seed500 alone and
# answered with a bare lemma sentence. The two are not interchangeable at
# inference: system prompt, target format and adapter all change together, so
# they are switched together here and nowhere else.
# Defaults to v2. Override per-shell rather than editing this line, so two
# people can work on different variants without conflicting in git:
#     PowerShell   $env:USE_REASONING = "0"
#     bash         USE_REASONING=0 python llm.py --train
USE_REASONING = os.environ.get("USE_REASONING", "1") != "0"

if USE_REASONING:
    import sources
    SYSTEM_PROMPT = skill.SYSTEM_PROMPT
    ADAPTER_DIR = skill.ADAPTER_DIR              # lemmatizer_lora_v2
    TRAIN_FILE = sources.OUT_TRAIN
    VAL_FILE = sources.OUT_VAL
    VAL_SEED_FILE = sources.OUT_VAL_SEED
    # Reasoning targets run ~150-300 tokens against v1's ~30, and the system
    # prompt is longer too (p95 target is 146 words). At 512 the longest rows
    # would train truncated, teaching the model to stop mid-derivation.
    # If this will not fit in 8GB, drop per_device_train_batch_size to 1 and
    # raise gradient_accumulation_steps to 8 rather than lowering this.
    MAX_SEQ_LEN = 1024
else:
    SYSTEM_PROMPT = skill.SYSTEM_PROMPT_V1
    # Own directory on purpose. `lemmatizer_lora` is shared, and two people
    # training into the same filename already cost one adapter (18 Aug run
    # overwritten by 20 Aug). Override per-run to keep experiments side by
    # side instead of overwriting the previous one:
    #     $env:ADAPTER_DIR = "lemmatizer_lora_yt8k"
    ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "lemmatizer_lora_yt")
    # The three-way design: crawl/build_silver.py --resplit owns these two,
    # and data/gold.jsonl is the test set, which nothing here may read.
    # The older training.jsonl / val_sft.jsonl are left in place as the record
    # of how the first lemmatizer_lora_yt was built; they are not read now.
    TRAIN_FILE = "data/training_split.jsonl"
    VAL_FILE = "data/dev_split.jsonl"
    # Was data/val_seed.jsonl, an orphan of the earlier seed500 run that has
    # now been removed. --eval scores the DEV split instead: gold.jsonl is the
    # test set and nothing in training may read it.
    VAL_SEED_FILE = "../data/dev_annotated.jsonl"
    MAX_SEQ_LEN = 512


# ----------------------------------------------------------------------------
# STAGE 1: PREPARE
# ----------------------------------------------------------------------------

def to_chatml(row):
    if USE_REASONING:
        return skill.to_chatml(row, SYSTEM_PROMPT)
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": row["text"]},
        {"role": "assistant", "content": row["lemma_text"]},
    ]}


def prepare():
    os.makedirs("data", exist_ok=True)

    # v2 composes its splits from the registry, so every linked corpus is
    # rebuilt from its own file. Nothing is merged on disk by hand.
    if USE_REASONING:
        sources.build()
        return

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

def _build(cls, **kw):
    """Construct `cls` with only the keyword arguments it actually accepts.

    This repo has moved across TRL and transformers versions, and the two
    renamed things underneath it: `evaluation_strategy` became `eval_strategy`
    in transformers 4.46, and SFTTrainer's `tokenizer` became
    `processing_class` in TRL 0.12. Rather than pin a version, ask the class
    what it takes. Anything dropped is printed, never silently discarded --
    a silently ignored eval_strategy is exactly how you end up believing you
    have validation when you do not.
    """
    import inspect
    from dataclasses import fields, is_dataclass

    if is_dataclass(cls):
        accepted = {f.name for f in fields(cls)}
    else:
        accepted = set(inspect.signature(cls.__init__).parameters)

    for new_name, old_name in (("eval_strategy", "evaluation_strategy"),
                               ("processing_class", "tokenizer")):
        if new_name in kw and new_name not in accepted and old_name in accepted:
            kw[old_name] = kw.pop(new_name)
        elif old_name in kw and old_name not in accepted and new_name in accepted:
            kw[new_name] = kw.pop(old_name)

    dropped = sorted(k for k in kw if k not in accepted)
    if dropped:
        print("[config] %s does not accept, so NOT applied: %s"
              % (cls.__name__, ", ".join(dropped)))
    return cls(**{k: v for k, v in kw.items() if k in accepted})


class TimeBudget:
    """Stop training once a wall-clock budget is spent.

    Subclassed off transformers.TrainerCallback at call time rather than at
    import time, so this module still imports on a machine with no GPU stack.

    Stopping early is safe here because save_steps == eval_steps and
    load_best_model_at_end is on: whatever the budget cuts short, the adapter
    saved is still the best checkpoint the dev set saw, not the last one.
    """

    @staticmethod
    def make(minutes):
        from transformers import TrainerCallback

        class _Cb(TrainerCallback):
            def on_train_begin(self, args, state, control, **kw):
                self.t0 = time.time()
                self.warned = False

            def on_step_end(self, args, state, control, **kw):
                spent = (time.time() - self.t0) / 60.0
                if not self.warned and spent > minutes * 0.8:
                    self.warned = True
                    print("\n[budget] %.0f of %d min spent at step %d/%d"
                          % (spent, minutes, state.global_step, state.max_steps))
                if spent >= minutes:
                    print("\n[budget] %d min reached at step %d/%d - stopping. "
                          "The best checkpoint so far is still what gets saved."
                          % (minutes, state.global_step, state.max_steps))
                    control.should_training_stop = True
                return control

        return _Cb()


def train(epochs=EPOCHS, budget_mins=BUDGET_MINS, batch=BATCH_SIZE,
          accum=ACCUM_STEPS, eval_rows=EVAL_ROWS, eval_every=EVAL_EVERY):
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

    # The dev split, which until now was written and then never read: train()
    # built only train_ds and passed no eval_dataset, so no validation loss
    # was computed at any point and val_sft.jsonl was a held-out split on
    # paper only. It is the signal that picks the checkpoint below.
    eval_ds = None
    if os.path.exists(VAL_FILE):
        eval_ds = load_dataset("json", data_files=VAL_FILE,
                               split="train").map(fmt, batched=True)
        full_dev = len(eval_ds)
        # Score a fixed subset in the loop. eval_loss is a mean over roughly
        # 15,000 target tokens at 200 rows, which is stable enough to rank
        # checkpoints; the full dev split is still what gets reported, via
        # `python crawl/evaluate.py --dev`. This is the single biggest saving
        # available and it cannot change the trained weights, because nothing
        # backpropagates through an evaluation.
        if eval_rows and full_dev > eval_rows:
            eval_ds = eval_ds.shuffle(seed=SEED).select(range(eval_rows))
            print("dev set: %d rows from %s, %d sampled for in-loop eval"
                  % (full_dev, VAL_FILE, eval_rows))
        else:
            print("dev set: %d rows from %s" % (full_dev, VAL_FILE))
    else:
        print("NO DEV SET at %s - training without validation." % VAL_FILE)

    print("\n" + "=" * 70)
    print(train_ds[0]["text"])
    print("=" * 70 + "\n")

    # Evaluate and checkpoint on the same cadence, which
    # load_best_model_at_end requires.
    steps_per_epoch = -(-len(train_ds) // (batch * accum))
    total_steps = steps_per_epoch * epochs
    print("\nplan: %d rows / (%d x %d) = %d steps an epoch x %d epochs "
          "= %d steps, %d evaluations, budget %s"
          % (len(train_ds), batch, accum, steps_per_epoch, epochs, total_steps,
             total_steps // eval_every,
             ("%d min" % budget_mins) if budget_mins else "none"))

    cfg = dict(
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        dataset_num_proc=1 if ON_WINDOWS else 2,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=accum,
        warmup_steps=5,
        num_train_epochs=epochs,
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
    )
    if eval_ds is not None:
        cfg.update(
            per_device_eval_batch_size=batch,
            eval_strategy="steps",
            eval_steps=eval_every,
            save_strategy="steps",
            save_steps=eval_every,
            save_total_limit=2,          # checkpoints are ~160 MB each
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )

    trainer = _build(
        SFTTrainer,
        model=model,
        processing_class=tokenizer,      # falls back to tokenizer= on old TRL
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=_build(SFTConfig, **cfg),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    # After train_on_responses_only, not before: it is documented to return
    # the trainer it was given, but attaching afterwards holds whether it
    # returns the same object or a rebuilt one.
    if budget_mins:
        trainer.add_callback(TimeBudget.make(budget_mins))

    stats = trainer.train()
    mins = stats.metrics["train_runtime"] / 60
    print("\nRuntime: %.1f min%s" % (
        mins, ("  (budget %d min)" % budget_mins) if budget_mins else ""))
    if trainer.state.global_step < trainer.state.max_steps:
        print("Stopped at step %d of %d - the budget cut the run short. "
              "Raise --budget-mins, or lower --epochs, if you want it to "
              "finish the schedule."
              % (trainer.state.global_step, trainer.state.max_steps))

    # Which checkpoint the dev set chose, and the curve it chose it from.
    hist = [h for h in trainer.state.log_history if "eval_loss" in h]
    if hist:
        best = min(hist, key=lambda h: h["eval_loss"])
        print("\ndev evaluations: %d" % len(hist))
        print("first  eval_loss %.4f at step %s"
              % (hist[0]["eval_loss"], hist[0].get("step")))
        print("last   eval_loss %.4f at step %s"
              % (hist[-1]["eval_loss"], hist[-1].get("step")))
        print("BEST   eval_loss %.4f at step %s  <- checkpoint kept"
              % (best["eval_loss"], best.get("step")))
        if best is hist[-1]:
            print("(best is the last step: the model was still improving, "
                  "so more epochs may help)")
    else:
        print("\nNo eval_loss recorded - check the [config] lines above.")

    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    # Record which skill this run trained on, in the adapter itself. chat.py
    # reads it back, so an adapter can never again be loaded with the wrong
    # system prompt because a shell happened to have USE_REASONING set
    # differently from the one that trained it.
    skill.write_version(ADAPTER_DIR, "v2" if USE_REASONING else "v1")
    print(f"Adapter saved to {ADAPTER_DIR}/ (skill {'v2' if USE_REASONING else 'v1'})")


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
    # Registry-linked rows carry no candidate lists; the token itself is then
    # the only baseline available, which is what `candidates` falls back to.
    cand = row.get("candidates") or {}
    return [(cand.get(e["t"]) or [e["t"]])[0] for e in row["lemmas"]]


def gold_rule(e):
    """The rule tag a row asserts, however that row records it."""
    if e.get("rule"):
        return e["rule"]
    return skill.rule_for(e.get("src"), e["t"], e["l"])[0]


def evaluate():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ADAPTER_DIR, max_seq_length=MAX_SEQ_LEN,
        dtype=None, load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template=CHAT_TEMPLATE)
    FastLanguageModel.for_inference(model)

    rows = [json.loads(l) for l in open(VAL_SEED_FILE, encoding="utf-8") if l.strip()]
    hits = defaultdict(lambda: {"model": 0, "base": 0, "n": 0, "rule": 0})
    exact, misaligned, malformed = 0, 0, 0
    by_source = defaultdict(lambda: {"model": 0, "n": 0})

    for row in rows:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["text"]}]
        ids = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
        # The derivation is far longer than v1's bare sentence; too small a cap
        # truncates mid-answer and looks like a format failure.
        out = model.generate(input_ids=ids,
                             max_new_tokens=400 if USE_REASONING else 160,
                             do_sample=False, pad_token_id=tokenizer.eos_token_id)
        pred = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

        # v2 answers carry their own token segmentation: one line per token, so
        # the predicted lemmas are read off directly instead of being re-aligned
        # by word count. That removes v1's worst failure mode, where a single
        # multi-word lemma shifted every later token and zeroed the whole row.
        rules = None
        if USE_REASONING:
            parsed = skill.parse_response(pred)
            if not parsed["ok"]:
                malformed += 1
            pred_text = parsed["lemma_text"]
            if len(parsed["steps"]) == len(row["lemmas"]):
                aligned = [s["lemma"] for s in parsed["steps"]]
                rules = [s["rule"] for s in parsed["steps"]]
            else:
                aligned = align(pred_text.split(), row["lemmas"])
        else:
            pred_text = pred
            aligned = align(pred_text.split(), row["lemmas"])

        # Metric 1: whole-sentence exact match
        gold_text = row.get("lemma_text") or " ".join(e["l"] for e in row["lemmas"])
        if norm(pred_text.replace(" ", "")) == norm(gold_text.replace(" ", "")):
            exact += 1

        if aligned is None:
            misaligned += 1

        # Metric 2: per-token, by difficulty category
        base = baseline_predict(row)
        for i, e in enumerate(row["lemmas"]):
            d = hits[e.get("src") or e.get("rule") or "?"]
            d["n"] += 1
            src_d = by_source[row.get("source", "?")]
            src_d["n"] += 1
            if aligned and norm(aligned[i]) == norm(e["l"]):
                d["model"] += 1
                src_d["model"] += 1
            if norm(base[i]) == norm(e["l"]):
                d["base"] += 1
            # Metric 3 (v2 only): did it cite the rule that actually applied?
            # A right lemma from the wrong rule is a lucky guess, and this is
            # the only metric that separates the two.
            if rules and rules[i] == gold_rule(e):
                d["rule"] += 1

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
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--budget-mins", type=int, default=BUDGET_MINS,
                   help="stop training after this many minutes (0 = no cap)")
    p.add_argument("--bs", type=int, default=BATCH_SIZE,
                   help="per-device batch; halve it and double --accum on OOM")
    p.add_argument("--accum", type=int, default=ACCUM_STEPS)
    p.add_argument("--eval-rows", type=int, default=EVAL_ROWS,
                   help="dev rows scored in the loop (0 = all of them)")
    p.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    a, _ = p.parse_known_args()          # tolerate Jupyter's -f kernel.json

    if a.prepare:
        prepare()
    elif a.train:
        train(epochs=a.epochs, budget_mins=a.budget_mins, batch=a.bs,
              accum=a.accum, eval_rows=a.eval_rows, eval_every=a.eval_every)
    elif a.eval:
        evaluate()
    else:
        p.print_help()