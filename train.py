#!/usr/bin/env python


import argparse
import json
import math
import os
import sys
import time
from pathlib import Path


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import skill                                                   # noqa: E402

TRAIN_FILE = ROOT / "unsloth" / "data" / "train_reasoning.jsonl"
DEV_FILE = ROOT / "unsloth" / "data" / "dev_reasoning.jsonl"



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
MAX_SEQ_LEN = 896        
EPOCHS = 1               
                 
BATCH_SIZE = 1           
                         
ACCUM_STEPS = 4
BUDGET_MINS = 120        
EVALS = 5               
                      
EVAL_ROWS = 64
LORA_RANK = 16
LEARNING_RATE = 2e-4


ATTN_ONLY = ["q_proj", "k_proj", "v_proj", "o_proj"]
ALL_LINEAR = ATTN_ONLY + ["gate_proj", "up_proj", "down_proj"]

PROBE_MICRO = 10         # micro-batches timed; a step is ACCUM_STEPS of them
PROBE_SKIP = 3           # first ones carry one-off kernel compilation
PROBE_MARGIN = 1.15      # the probe runs before the optimiser state exists, so
                         
                         
SAVE_RESERVE_MIN = 4     




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


class TimeBudget:


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

        return _Cb()


# ---------------------------------------------------------------------------
# HOW LONG A STEP ACTUALLY TAKES
# ---------------------------------------------------------------------------

def _probe_step_seconds(trainer, accum):
   
    import statistics

    import torch

    model = trainer.model
    device = next(model.parameters()).device
    loader = iter(trainer.get_train_dataloader())

    micro, toks = [], []
    was_training = model.training
    model.train()
    try:
        for _ in range(PROBE_MICRO):
            batch = next(loader)
            batch = {k: (v.to(device) if hasattr(v, "to") else v)
                     for k, v in batch.items()}
            ids = batch.get("input_ids")
            if ids is not None:
                toks.append(int(ids.numel()))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()
            (model(**batch).loss / accum).backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            micro.append(time.time() - t0)
    finally:
        model.zero_grad(set_to_none=True)
        if not was_training:
            model.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Median, not mean: one micro-batch colliding with a Windows compositor
    # hitch should not resize the whole plan. The first few carry one-off
    # kernel compilation and are dropped.
    kept = micro[PROBE_SKIP:] or micro
    per_step = statistics.median(kept) * accum * PROBE_MARGIN
    mean_tok = (sum(toks) / len(toks) / trainer.args.per_device_train_batch_size
                if toks else 625.0)
    return per_step, mean_tok


def _plan_steps(sec_per_step, mean_tok, batch, accum, ceiling, budget_mins,
                spent_s, evals, eval_rows):
    """Decide how many steps fit, and show the arithmetic that decided it."""
    tok_per_s = (batch * accum * mean_tok) / sec_per_step
    # Evaluation is forward-only: no backward, no activation recompute. Three
    # times training throughput is the conservative end of that; over-charging
    # for eval costs a few steps, under-charging overruns the budget.
    eval_s = (evals * eval_rows * mean_tok) / (tok_per_s * 3.0)
    left_s = budget_mins * 60 - spent_s - SAVE_RESERVE_MIN * 60
    affordable = int((left_s - eval_s) // sec_per_step)
    chosen = max(1, min(affordable, ceiling))

    print("\n--- budget plan " + "-" * 47)
    print("  measured        %.2f s/step  (%.0f tokens/s, %.0f tok/row)"
          % (sec_per_step, tok_per_s, mean_tok))
    print("  budget          %d min, %.1f already spent, %d held for saving"
          % (budget_mins, spent_s / 60, SAVE_RESERVE_MIN))
    print("  %d dev evals    ~%.1f min" % (evals, eval_s / 60))
    print("  affordable      %d steps%s"
          % (affordable, "" if affordable <= ceiling
             else " (capped at %d = %d epoch(s))" % (ceiling, EPOCHS)))
    print("  RUNNING         %d steps = %s examples = %.0f%% of an epoch"
          % (chosen, format(chosen * batch * accum, ","),
             100.0 * chosen / ceiling * EPOCHS))
    print("  projected       %.0f min of stepping, %.0f min total"
          % (chosen * sec_per_step / 60,
             (spent_s + chosen * sec_per_step + eval_s) / 60 + SAVE_RESERVE_MIN))
    print("-" * 62)

    if affordable < 0.25 * ceiling:
        print("\n  ** %.0f%% of an epoch is thin. The card is the bottleneck,"
              % (100.0 * affordable / ceiling * EPOCHS))
        print("     not the schedule. In order of effect:")
        print("       1. run this on a free Kaggle/Colab T4 -- 16 GB, no")
        print("          spilling, roughly 10x these numbers")
        print("       2. --lora-attn-only          (24.3M -> 9.2M trained)")
        print("       3. turn OFF 'CUDA - Sysmem Fallback Policy' for")
        print("          python.exe in the NVIDIA Control Panel, so a config")
        print("          that does not fit says so instead of quietly paging")
        print("       4. close anything else holding VRAM; drive the display")
        print("          from the iGPU\n")
    return chosen


# ---------------------------------------------------------------------------
# TRAIN
# ---------------------------------------------------------------------------

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

    # The derivation targets are roughly half of each rendered example, so a
    # mask covering nearly everything means the prompt is being trained on too.
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
    path = ROOT / "unsloth" / spec["adapter"] / RUN_META
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
    other_dir = ROOT / "unsloth" / MODELS[other]["adapter"] / RUN_META
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
        print("[fairness] This run has no --max-steps, so the probe will pick")
        print("           its own length from the clock and the two models will")
        print("           almost certainly differ. To compare pretraining "
              "rather than speed:")
        print("               --max-steps %d\n" % meta["steps"])
    elif requested_steps != meta["steps"]:
        print("[fairness] --max-steps %d does not match. The comparison will "
              "confound\n           pretraining with schedule.\n"
              % requested_steps)
    else:
        print("[fairness] matches - the two runs are comparable.\n")


def train(spec, epochs, budget_mins, batch, accum, evals, eval_rows,
          max_steps, attn_only, dry_run, verify):
    # The budget covers everything from here: loading the model, tokenising,
    # probing, training, evaluating and saving. Starting the clock at
    # trainer.train() would be measuring the wrong thing.
    t_start = time.time()
    for path in (TRAIN_FILE, DEV_FILE):
        if not path.exists():
            sys.exit("missing %s - run `python crawl/build_sft.py` first"
                     % path.relative_to(ROOT))

    adapter_dir = ROOT / "unsloth" / spec["adapter"]
    print("base      %s" % spec["base"])
    print("template  %s" % spec["template"])
    print("adapter   %s" % adapter_dir.relative_to(ROOT))
    print("data      %s / %s" % (TRAIN_FILE.name, DEV_FILE.name))
    print("effective batch %d  (%d x %d accum)" % (batch * accum, batch, accum))
    print("seq len   %d | LoRA r%d on %s" % (
        MAX_SEQ_LEN, LORA_RANK,
        "q/k/v/o only" if attn_only else "all 7 projections"))
    print("budget    %s\n" % ("%d min" % budget_mins if budget_mins else "none"))
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

    run_dir = ROOT / "unsloth" / "training_runs" / spec["adapter"]
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
        print("--max-steps given: %d steps, no probe.\n" % max_steps)
    elif not budget_mins:
        max_steps = ceiling
        print("no budget: the full %d epoch(s), %d steps.\n"
              % (epochs, max_steps))
    else:
        print("probing %d micro-batches to time this configuration...\n"
              % PROBE_MICRO)
        try:
            sec, mean_tok = _probe_step_seconds(trainer, accum)
            max_steps = _plan_steps(sec, mean_tok, batch, accum, ceiling,
                                    budget_mins, time.time() - t_start,
                                    evals, eval_rows)
        except Exception as exc:                                # noqa: BLE001
          
            max_steps = min(ceiling, 300)
            print("\n[probe] failed (%s: %s)" % (type(exc).__name__, exc))
            print("[probe] falling back to %d steps; TimeBudget still caps "
                  "the run at %d min.\n" % (max_steps, budget_mins))


    eval_every = min(max_steps, max(20, max_steps // max(1, evals)))
    trainer.args.max_steps = max_steps
    trainer.args.eval_steps = eval_every
    trainer.args.save_steps = eval_every
   
    if budget_mins:
        trainer.add_callback(TimeBudget.make(budget_mins))

    stats = trainer.train()

    # ---- what the dev set chose -----------------------------------------
    print("\nRuntime: %.1f min" % (stats.metrics["train_runtime"] / 60))
    if trainer.state.global_step < trainer.state.max_steps:
        print("Stopped at step %d of %d - the budget cut the run short."
              % (trainer.state.global_step, trainer.state.max_steps))

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
        if best is hist[-1]:
            print("(best is the last step: eval loss was still falling when the")
            print(" budget ran out. Nothing overfitted; there was simply more")
            print(" training left to do than the clock allowed.)")
    else:
        print("\nNo eval_loss recorded - check the [config] lines above.")

    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    # Stamp the skill into the adapter, so it can never be served a prompt it
    # was not trained on.
    skill.write_version(str(adapter_dir))

   
    write_run_meta(spec, trainer.state.global_step, batch, accum, best_loss)

    print("\nAdapter saved to %s/ (skill %s)"
          % (adapter_dir.relative_to(ROOT), skill.SKILL_VERSION))
    print("\nTrain the other family to the SAME step count, not the same clock:")
    print("    python train.py --model %s --max-steps %d"
          % ("qwen" if spec["adapter"].endswith("llama") else "llama",
             trainer.state.global_step))
    print("\nTry it:  python main.py --model %s" % adapter_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[1],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=sorted(MODELS),
                   help="which family to fine-tune")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--bs", type=int, default=BATCH_SIZE,
                   help="per-device batch (default %d)" % BATCH_SIZE)
    p.add_argument("--accum", type=int, default=ACCUM_STEPS,
                   help="gradient accumulation (default %d)" % ACCUM_STEPS)
    p.add_argument("--budget-mins", type=int, default=BUDGET_MINS,
                   help="wall-clock cap, 0 to disable (default %d)" % BUDGET_MINS)
    p.add_argument("--evals", type=int, default=EVALS,
                   help="dev evaluations spread across the run (default %d)"
                        % EVALS)
    p.add_argument("--max-steps", type=int, default=0,
                   help="fix the run length and skip the probe")
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
    train(MODELS[a.model], a.epochs, a.budget_mins, a.bs, a.accum,
          a.evals, a.eval_rows, a.max_steps, a.lora_attn_only, a.dry_run,
          a.verify)
