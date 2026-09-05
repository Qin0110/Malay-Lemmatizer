import argparse
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "unsloth" / "training_runs"

TRAIN_COLOURS = {"llama": "#93c5fd", "qwen": "#fcd34d"}
DEV_COLOURS = {"llama": "#dc2626", "qwen": "#059669"}
FALLBACK_TRAIN, FALLBACK_DEV = "#94a3b8", "#0f172a"


def latest_state(adapter_dir):
    """Newest trainer_state.json under a run dir; works mid-run."""
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
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=140)

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

            ax.plot([best_step, best_step], [best_loss, 0],
                    color=dc, lw=0.8, ls=":", alpha=0.5, zorder=1)
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
    ax.set_ylim(bottom=0)   # only now, so autoscaling has seen the data range
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
    p = argparse.ArgumentParser()
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
