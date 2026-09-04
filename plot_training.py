#!/usr/bin/env python
"""
Chart the training and development loss curves.

    python plot_training.py                      # every model trained so far
    python plot_training.py --model llama        # just one
    python plot_training.py --out figures/       # where to write the PNG
    python plot_training.py --table              # print the numbers instead

WHAT THIS PLOTS, AND WHAT IT CANNOT
-----------------------------------
Loss, not accuracy. Fine-tuning optimises cross-entropy over next-token
prediction, so loss is the only quantity the training loop ever computes.
Accuracy for this task means generating a full derivation and comparing every
lemma against the gold answer, which is a generation pass rather than a
forward pass -- roughly twenty times the cost. Measuring it at each evaluation
would consume more of the budget than the training itself.

So the division of labour is:

    this script        loss curves, from the training run       (free)
    crawl/evaluate.py  accuracy, F1, WER, CER, McNemar          (once, at the end)

Both belong in the report and they answer different questions. The loss curve
shows whether the run was still learning when it stopped -- which, on a budget
this tight, is the thing a reader will want to know. The evaluation table shows
whether the result is any good.

READING THE CURVE
-----------------
Training loss falling while dev loss rises is overfitting, and the checkpoint
selection has already handled it: the adapter saved is the one at the lowest
dev loss, not the last one. Both still falling at the right-hand edge means
the budget ran out before the model did, which is the expected shape here.
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "unsloth" / "training_runs"

# Train and dev get genuinely different hues, not a light/dark pair of the
# same one -- a shade difference reads as "the same line, slightly faded" at
# a glance, which is the opposite of what the dev curve needs: it is the
# number that actually matters (it is what picks the checkpoint), so it has
# to be the color the eye lands on first.
TRAIN_COLOURS = {"llama": "#93c5fd", "qwen": "#fcd34d"}   # pale, recedes
DEV_COLOURS = {"llama": "#dc2626", "qwen": "#059669"}     # bold, leads
FALLBACK_TRAIN, FALLBACK_DEV = "#94a3b8", "#0f172a"


def latest_state(adapter_dir):
    """The newest trainer_state.json under a run directory, or None.

    Read from the checkpoints rather than from a summary file, so this works
    on a run that is still in progress -- which is the moment you most want to
    see the curve.
    """
    states = glob.glob(str(adapter_dir / "checkpoint-*" / "trainer_state.json"))
    if not states:
        return None
    newest = max(states, key=os.path.getmtime)
    try:
        with open(newest, encoding="utf-8") as fh:
            return json.load(fh), newest
    except (OSError, ValueError):
        return None


def series(state):
    """(train points, dev points) as [(step, loss), ...]."""
    train, dev = [], []
    for e in state.get("log_history", []):
        if "loss" in e and "eval_loss" not in e:
            train.append((e["step"], e["loss"]))
        elif "eval_loss" in e:
            dev.append((e["step"], e["eval_loss"]))
    return train, dev


def collect(models):
    out = {}
    for name in models:
        found = latest_state(RUNS / ("lemmatizer_lora_%s" % name))
        if not found:
            continue
        state, path = found
        train, dev = series(state)
        if train or dev:
            out[name] = {"train": train, "dev": dev, "state": state,
                         "path": path}
    return out


def print_table(runs):
    for name, r in runs.items():
        s = r["state"]
        print("\n%s  --  step %d of %d (epoch %.3f)"
              % (name.upper(), s["global_step"], s["max_steps"], s["epoch"]))
        print("  source: %s" % os.path.relpath(r["path"], ROOT))
        print("  %-8s %-12s %-12s" % ("step", "train loss", "dev loss"))
        merged = {}
        for st, v in r["train"]:
            merged.setdefault(st, [None, None])[0] = v
        for st, v in r["dev"]:
            merged.setdefault(st, [None, None])[1] = v
        for st in sorted(merged):
            tr, dv = merged[st]
            print("  %-8d %-12s %-12s"
                  % (st,
                     "%.4f" % tr if tr is not None else "",
                     "%.4f" % dv if dv is not None else ""))
        if r["dev"]:
            best = min(r["dev"], key=lambda x: x[1])
            print("  best dev loss %.4f at step %d%s"
                  % (best[1], best[0],
                     "  (the last one - still improving)"
                     if best == r["dev"][-1] else ""))


def plot(runs, out_dir):
    import matplotlib
    matplotlib.use("Agg")            # no display on a headless box
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=140)

    # Placed above the point for the first run, below for the second, so two
    # callout boxes near the same step never sit on top of each other.
    box_side = 1
    for name, r in runs.items():
        tc = TRAIN_COLOURS.get(name, FALLBACK_TRAIN)
        dc = DEV_COLOURS.get(name, FALLBACK_DEV)

        if r["train"]:
            ax.plot([s for s, _ in r["train"]], [v for _, v in r["train"]],
                    color=tc, lw=1.6, alpha=0.9, zorder=2,
                    label="%s  train" % name)

        if r["dev"]:
            xs = [s for s, _ in r["dev"]]
            ys = [v for _, v in r["dev"]]
            best_step, best_loss = min(r["dev"], key=lambda x: x[1])
            ax.plot(xs, ys, color=dc, lw=2.2, ls="--", marker="o", ms=4.5,
                    mfc="white", mew=1.4, zorder=4,
                    label="%s  dev" % name)

            # A light drop-line anchors the best point to the x-axis (fixed at
            # y=0 below, not read back from get_ylim -- asking for the current
            # limit here, before the axis has seen all the data, would freeze
            # autoscaling at whatever placeholder range an empty axes starts
            # with, and every curve above it would be clipped off).
            ax.plot([best_step, best_step], [best_loss, 0],
                    color=dc, lw=0.8, ls=":", alpha=0.5, zorder=1)

            # A plain filled circle, sized up from the ordinary dev markers
            # rather than swapped for a different shape -- the point is to
            # emphasise the same series, not to introduce a new symbol.
            ax.scatter([best_step], [best_loss], marker="o", s=95,
                      facecolors=dc, edgecolors="black", linewidths=1.0,
                      zorder=6)

            ax.annotate(
                "%s best: %.4f" % (name, best_loss),
                (best_step, best_loss),
                textcoords="offset points",
                xytext=(0, 15 * box_side - 6),
                ha="center",
                fontsize=8.5, color="#1f2937",
                bbox=dict(boxstyle="square,pad=0.3", facecolor="white",
                         edgecolor="#94a3b8", linewidth=0.8),
                zorder=7,
            )
            box_side *= -1

    ax.set_xlabel("Optimiser step", fontsize=10.5)
    ax.set_ylabel("Cross-entropy loss", fontsize=10.5)
    ax.set_title("Fine-tuning loss (solid = train, dashed = dev)", fontsize=11)
    ax.tick_params(labelsize=9.5)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_ylim(bottom=0)   # set only now: earlier would freeze autoscaling
                            # before the top had seen the real data range
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=True, fontsize=9.5, loc="upper right",
             facecolor="white", edgecolor="#cbd5e1", framealpha=0.95)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ("training_loss_%s.png" % "_".join(sorted(runs)))
    fig.savefig(path)
    plt.close(fig)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--model", action="append", choices=("llama", "qwen"),
                   help="restrict to one family (repeatable)")
    p.add_argument("--out", default="figures", help="output directory")
    p.add_argument("--table", action="store_true",
                   help="print the numbers instead of plotting")
    a = p.parse_args()

    runs = collect(a.model or ["llama", "qwen"])
    if not runs:
        sys.exit("No trainer_state.json found under %s - has training run?"
                 % RUNS.relative_to(ROOT))

    for name, r in runs.items():
        s = r["state"]
        print("%-6s step %d/%d, %d dev evaluations"
              % (name, s["global_step"], s["max_steps"], len(r["dev"])))

    if a.table:
        print_table(runs)
        return
    path = plot(runs, ROOT / a.out)
    print("\nwrote %s" % path.relative_to(ROOT))
    if len(runs) == 1:
        print("Run again once both models are trained to get them on one axis.")


if __name__ == "__main__":
    main()
