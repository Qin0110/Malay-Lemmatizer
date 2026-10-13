#!/usr/bin/env python
"""Fine-tune Llama 3.2 3B or Qwen2.5 3B with QLoRA on data/sft/.

    python train.py --model llama --dry-run     show the settings, no GPU
    python train.py --model llama --verify      check the loss mask and stop
    python train.py --model llama               one epoch -> models/lemmatizer_lora_llama/
    python train.py --model qwen --max-steps N

Out of memory? Try --bs 1 --accum 4.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import skill                                                   # noqa: E402

MODELS_DIR = ROOT / "models"
TRAIN_FILE = ROOT / "data" / "sft" / "train.jsonl"
DEV_FILE = ROOT / "data" / "sft" / "dev.jsonl"


MODELS = {
    "llama": {
        "base": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        "template": "llama-3.2",
        "instruction_part": "<|start_header_id|>user<|end_header_id|>\n\n",
        "response_part": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "adapter": "lemmatizer_lora_llama",
    },
    "qwen": {
        "base": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
        "template": "qwen-2.5",
        "instruction_part": "<|im_start|>user\n",
        "response_part": "<|im_start|>assistant\n",
        "adapter": "lemmatizer_lora_qwen",
    },
}


SEED = 3407
MAX_SEQ_LEN = 1024
EPOCHS = 1

BATCH_SIZE = 4
ACCUM_STEPS = 1
EVALS = 5
EVAL_ROWS = 64
LORA_RANK = 16
LEARNING_RATE = 2e-4


ATTN_ONLY = ["q_proj", "k_proj", "v_proj", "o_proj"]
ALL_LINEAR = ATTN_ONLY + ["gate_proj", "up_proj", "down_proj"]


def _build(cls, **kw):
    import inspect
    from dataclasses import fields, is_dataclass

    if is_dataclass(cls):
        accepted = {f.name for f in fields(cls)}
    else:
        accepted = set(inspect.signature(cls.__init__).parameters)

    for new, old in (("eval_strategy", "evaluation_strategy"),
                     ("processing_class", "tokenizer")):
        if new in kw and new not in accepted and old in accepted:
            kw[old] = kw.pop(new)
        elif old in kw and old not in accepted and new in accepted:
            kw[new] = kw.pop(old)

    dropped = sorted(k for k in kw if k not in accepted)
    if dropped:
        print("[config] %s does not accept, so NOT applied: %s"
              % (cls.__name__, ", ".join(dropped)))
    return cls(**{k: v for k, v in kw.items() if k in accepted})


def _verify_mask(trainer, tokenizer, verbose):
    batch = trainer.data_collator([trainer.train_dataset[0]])
    labels = batch["labels"][0]
    ids = batch["input_ids"][0]

    kept = [i for i, v in enumerate(labels.tolist()) if v != -100]
    total = len(labels)
    if not kept:
        sys.exit("MASK CHECK FAILED: every token is masked out, so the loss "
                 "has nothing to train on. The response marker did not match.")

    supervised = tokenizer.decode([ids[i] for i in kept],
                                  skip_special_tokens=False)
    prompt_head = tokenizer.decode(ids[:kept[0]].tolist(),
                                   skip_special_tokens=False)

    pct = 100.0 * len(kept) / total
    print("loss mask: %d of %d tokens supervised (%.0f%%)" % (len(kept), total, pct))

    if pct > 90:
        sys.exit("MASK CHECK FAILED: %.0f%% of tokens are supervised. The "
                 "instruction marker did not match, so the model is being "
                 "trained on the system prompt as well as the answer." % pct)
    if "<lema>" not in supervised:
        sys.exit("MASK CHECK FAILED: the supervised span does not contain "
                 "<lema>, so it is not the answer.\nsupervised span begins: %r"
                 % supervised[:200])
    if "Anda pelemma" in supervised:
        sys.exit("MASK CHECK FAILED: the system prompt is inside the "
                 "supervised span.")
    print("loss mask covers the answer, not the prompt   PASS")

    if verbose:
        print("\n--- mask detail " + "-" * 46)
        print("  NOT trained on (ends): ...%s" % prompt_head[-160:].replace("\n", "\\n"))
        print("  TRAINED on (begins):   %s..." % supervised[:160].replace("\n", "\\n"))
        print("-" * 62)
        print("\n--verify: setup confirmed, exiting before training.")
        sys.exit(0)


RUN_META = "run_meta.json"


def write_run_meta(spec, steps, batch, accum, best_loss):
    """Record the schedule this run actually got, beside the adapter."""
    path = MODELS_DIR / spec["adapter"] / RUN_META
    path.write_text(json.dumps({
        "base": spec["base"],
        "steps": steps,
        "batch": batch,
        "accum": accum,
        "examples": steps * batch * accum,
        "max_seq_len": MAX_SEQ_LEN,
        "lora_rank": LORA_RANK,
        "best_eval_loss": best_loss,
    }, indent=2), encoding="utf-8")
    return path


def _check_fairness(spec, requested_steps):
    other = "qwen" if spec["adapter"].endswith("llama") else "llama"
    other_dir = MODELS_DIR / MODELS[other]["adapter"] / RUN_META
    if not other_dir.exists():
        return
    try:
        meta = json.loads(other_dir.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return

    print("[fairness] %s was trained for %d steps (%s examples, batch %dx%d)."
          % (other, meta["steps"], format(meta["examples"], ","),
             meta["batch"], meta["accum"]))
    # 0 is the argparse default and means "not given", the same as None.
    if not requested_steps:
        print("[fairness] This run has no --max-steps, so it runs a full epoch.")
        print("           Keep --bs and --accum the same as that run, or pin")
        print("           the length with:")
        print("               --max-steps %d\n" % meta["steps"])
    elif requested_steps != meta["steps"]:
        print("[fairness] --max-steps %d does not match. The comparison will "
              "confound\n           pretraining with schedule.\n"
              % requested_steps)
    else:
        print("[fairness] matches - the two runs are comparable.\n")


def train(spec, epochs, batch, accum, evals, eval_rows, max_steps, attn_only,
          dry_run, verify):
    for path in (TRAIN_FILE, DEV_FILE):
        if not path.exists():
            sys.exit("missing %s - run `python crawl/build_sft.py` first"
                     % path.relative_to(ROOT))

    adapter_dir = MODELS_DIR / spec["adapter"]
    print("base      %s" % spec["base"])
    print("template  %s" % spec["template"])
    print("adapter   %s" % adapter_dir.relative_to(ROOT))
    print("data      %s / %s" % (TRAIN_FILE.name, DEV_FILE.name))
    print("effective batch %d  (%d x %d accum)" % (batch * accum, batch, accum))
    print("seq len   %d | LoRA r%d on %s\n" % (
        MAX_SEQ_LEN, LORA_RANK,
        "q/k/v/o only" if attn_only else "all 7 projections"))
    _check_fairness(spec, max_steps)
    if dry_run:
        print("--dry-run: no model loaded.")
        return

    from datasets import load_dataset
    from unsloth import is_bfloat16_supported
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template, train_on_responses_only

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=spec["base"],
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=(ATTN_ONLY if attn_only else ALL_LINEAR),
        lora_alpha=LORA_RANK,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )
    tokenizer = get_chat_template(tokenizer, chat_template=spec["template"])

    def fmt(examples):
        return {"text": [
            tokenizer.apply_chat_template(m, tokenize=False,
                                          add_generation_prompt=False)
            for m in examples["messages"]
        ]}

    train_ds = load_dataset("json", data_files=str(TRAIN_FILE),
                            split="train").map(fmt, batched=True)
    eval_ds = load_dataset("json", data_files=str(DEV_FILE),
                           split="train").map(fmt, batched=True)
    full_dev = len(eval_ds)
    if eval_rows and eval_rows < full_dev:

        eval_ds = eval_ds.shuffle(seed=SEED).select(range(eval_rows))


    sample = train_ds[0]["text"]
    for name in ("instruction_part", "response_part"):
        if spec[name] not in sample:
            print("\nRendered sample begins:\n%s\n" % sample[:400])
            sys.exit("%s %r does not appear in the rendered chat text. The "
                     "loss mask would silently not apply and the model would "
                     "train on the prompt. Fix MODELS[%r] before training."
                     % (name, spec[name], spec["adapter"]))
    print("loss-mask markers found in the rendered text  PASS")
    print("train %d rows | dev %d rows\n" % (len(train_ds), len(eval_ds)))

    run_dir = MODELS_DIR / "training_runs" / spec["adapter"]
    ceiling = math.ceil(len(train_ds) / (batch * accum)) * epochs


    cfg = dict(
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=accum,
        warmup_steps=0,
        warmup_ratio=0.03,
        max_steps=ceiling,
        learning_rate=LEARNING_RATE,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        seed=SEED,
        output_dir=str(run_dir),
        report_to="none",
        max_seq_length=MAX_SEQ_LEN,
        per_device_eval_batch_size=batch,
        eval_strategy="steps",
        eval_steps=ceiling,
        save_strategy="steps",
        save_steps=ceiling,
        save_total_limit=2,

        save_only_model=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = _build(
        SFTTrainer,
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=_build(SFTConfig, **cfg),
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part=spec["instruction_part"],
        response_part=spec["response_part"],
    )
    _verify_mask(trainer, tokenizer, verify)


    if max_steps:
        max_steps = min(max_steps, ceiling)
        print("--max-steps given: %d steps.\n" % max_steps)
    else:
        max_steps = ceiling
        print("the full %d epoch(s), %d steps.\n" % (epochs, max_steps))

    eval_every = min(max_steps, max(20, max_steps // max(1, evals)))
    trainer.args.max_steps = max_steps
    trainer.args.eval_steps = eval_every
    trainer.args.save_steps = eval_every

    stats = trainer.train()

    print("\nRuntime: %.1f min" % (stats.metrics["train_runtime"] / 60))

    hist = [h for h in trainer.state.log_history if "eval_loss" in h]
    best_loss = None
    if hist:
        best = min(hist, key=lambda h: h["eval_loss"])
        best_loss = best["eval_loss"]
        print("\ndev evaluations: %d" % len(hist))
        print("first  eval_loss %.4f at step %s" % (hist[0]["eval_loss"],
                                                    hist[0].get("step")))
        print("last   eval_loss %.4f at step %s" % (hist[-1]["eval_loss"],
                                                    hist[-1].get("step")))
        print("BEST   eval_loss %.4f at step %s  <- checkpoint kept"
              % (best["eval_loss"], best.get("step")))
    else:
        print("\nNo eval_loss recorded - check the [config] lines above.")

    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    # stamp the skill version so the adapter cannot be served a prompt it
    # was not trained on
    skill.write_version(str(adapter_dir))


    write_run_meta(spec, trainer.state.global_step, batch, accum, best_loss)

    print("\nAdapter saved to %s/ (skill %s)"
          % (adapter_dir.relative_to(ROOT), skill.SKILL_VERSION))
    print("\nTrain the other family to the SAME step count:")
    print("    python train.py --model %s --max-steps %d"
          % ("qwen" if spec["adapter"].endswith("llama") else "llama",
             trainer.state.global_step))
    print("\nTry it:  python main.py --model %s-ft" % spec["adapter"].rsplit("_", 1)[-1])


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=sorted(MODELS),
                   help="which family to fine-tune")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--bs", type=int, default=BATCH_SIZE,
                   help="per-device batch (default %d)" % BATCH_SIZE)
    p.add_argument("--accum", type=int, default=ACCUM_STEPS,
                   help="gradient accumulation (default %d)" % ACCUM_STEPS)
    p.add_argument("--evals", type=int, default=EVALS,
                   help="dev evaluations spread across the run (default %d)"
                        % EVALS)
    p.add_argument("--max-steps", type=int, default=0,
                   help="stop early instead of running the full epoch")
    p.add_argument("--verify", action="store_true",
                   help="check the loss mask on a real batch and exit "
                        "(~1 min, no training)")
    p.add_argument("--lora-attn-only", action="store_true",
                   help="LoRA on q/k/v/o only: 9.2M trained, not 24.3M")
    p.add_argument("--eval-rows", type=int, default=EVAL_ROWS,
                   help="dev rows used in-loop, 0 for all (default %d)"
                        % EVAL_ROWS)
    p.add_argument("--dry-run", action="store_true",
                   help="print the config and stop, no GPU needed")
    a = p.parse_args()
    train(MODELS[a.model], a.epochs, a.bs, a.accum, a.evals, a.eval_rows,
          a.max_steps, a.lora_attn_only, a.dry_run, a.verify)
