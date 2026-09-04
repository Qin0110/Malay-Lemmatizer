#!/usr/bin/env python
"""
Import adapters trained elsewhere (Colab, a lab machine) into this project.

    python import_adapters.py --from "D:/Drive/lemmatizer_adapters"
    python import_adapters.py --from ... --dry-run
    python import_adapters.py --from ... --keep-existing   # do not overwrite

WHAT THIS IS FOR
----------------
Training somewhere with a bigger GPU and finishing here is the normal shape of
this project: a 4 GB laptop card completes about 11% of an epoch in two hours,
where a 16 GB T4 completes the whole thing. What comes back is two adapter
folders and a tree of trainer_state.json files, and three things have to be
true before they can be used or reported:

  1. Both models must carry the SAME step count. Step count is training-data
     volume, so unequal steps would mean the comparison measured which model
     trained longer rather than which pretraining suits Malay. Asserted here
     rather than discovered later in the results table.

  2. Each adapter must carry its skill stamp, or inference will serve it a
     prompt it was never trained on -- which fails silently, producing fluent
     output in the wrong format.

  3. `best_model_checkpoint` inside each trainer_state.json points at wherever
     the training ran. After copying, that directory does not exist on this
     machine, so the pointer is stale and is repointed at the copy. This is
     housekeeping on a path field, not a claim about provenance: record the
     hardware you actually trained on in the report.

Nothing in this project reads that pointer, so leaving it stale breaks
nothing. It is fixed so the files are self-consistent with where they live.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEST_ADAPTERS = ROOT / "unsloth"
DEST_RUNS = ROOT / "unsloth" / "training_runs"
FAMILIES = ("llama", "qwen")


def find_source(src, name):
    """The adapter folder for one family, wherever it sits under `src`."""
    want = "lemmatizer_lora_%s" % name
    direct = src / want
    if (direct / "adapter_model.safetensors").exists():
        return direct
    for cand in src.rglob(want):
        if (cand / "adapter_model.safetensors").exists():
            return cand
    return None


def repoint(state_path, new_dir):
    """Repoint a stale best_model_checkpoint at the copy on this machine."""
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    old = state.get("best_model_checkpoint")
    if not old:
        return None
    # Keep the checkpoint's own folder name; only the parent path moved.
    state["best_model_checkpoint"] = str(new_dir / Path(old.replace("\\", "/")).name)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return old, state["best_model_checkpoint"]


def main(src, dry_run, keep_existing):
    if not src.exists():
        sys.exit("No such folder: %s" % src)

    found, metas = {}, {}
    for name in FAMILIES:
        path = find_source(src, name)
        if path is None:
            print("  %-6s NOT FOUND under %s" % (name, src))
            continue
        found[name] = path
        meta_path = path / "run_meta.json"
        if meta_path.exists():
            try:
                metas[name] = json.loads(meta_path.read_text(encoding="utf-8"))
            except ValueError:
                pass
        print("  %-6s found  %s" % (name, path))

    if not found:
        sys.exit("Nothing to import.")

    # ---- the assertion the comparison rests on ---------------------------
    print("\nFAIRNESS")
    steps = {n: m.get("steps") for n, m in metas.items()}
    for name in found:
        m = metas.get(name)
        if not m:
            print("  %-6s no run_meta.json - cannot verify its schedule" % name)
        else:
            print("  %-6s %d steps, %s examples, batch %dx%d"
                  % (name, m["steps"], format(m["examples"], ","),
                     m["batch"], m["accum"]))
    distinct = {s for s in steps.values() if s is not None}
    if len(distinct) > 1:
        print("\n  ** STEP COUNTS DIFFER: %s" % steps)
        print("     The two models did not see the same amount of data, so a")
        print("     comparison between them would confound pretraining with")
        print("     schedule. Retrain the shorter one with --max-steps %d"
              % max(distinct))
        if not dry_run:
            sys.exit("\nRefusing to import a mismatched pair. "
                     "Re-run with --dry-run to inspect, or fix the short run.")
    elif len(distinct) == 1:
        print("  matched at %d steps - comparable." % distinct.pop())

    # ---- skill stamp ------------------------------------------------------
    print("\nSKILL STAMP")
    for name, path in found.items():
        stamp = path / "skill.json"
        if stamp.exists():
            v = json.loads(stamp.read_text(encoding="utf-8")).get("skill_version")
            print("  %-6s %s" % (name, v))
        else:
            print("  %-6s MISSING - inference would guess the prompt" % name)

    if dry_run:
        print("\n--dry-run: nothing copied.")
        return

    # ---- copy -------------------------------------------------------------
    print("\nCOPYING")
    for name, path in found.items():
        dest = DEST_ADAPTERS / ("lemmatizer_lora_%s" % name)
        if dest.exists():
            if keep_existing:
                print("  %-6s skipped, already present" % name)
                continue
            backup = dest.with_name(dest.name + "_prev")
            shutil.rmtree(backup, ignore_errors=True)
            dest.rename(backup)
            print("  %-6s existing -> %s" % (name, backup.name))
        shutil.copytree(path, dest)
        print("  %-6s -> %s" % (name, dest.relative_to(ROOT)))

    # ---- trainer_state trees, for the loss curves -------------------------
    print("\nTRAINING RUNS (loss curves)")
    for name in found:
        run_name = "lemmatizer_lora_%s" % name
        srcs = list(src.rglob("%s/checkpoint-*/trainer_state.json" % run_name))
        if not srcs:
            print("  %-6s none found - plot_training.py will have no curve"
                  % name)
            continue
        for s in srcs:
            out = DEST_RUNS / run_name / s.parent.name
            out.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, out / "trainer_state.json")
            moved = repoint(out / "trainer_state.json", DEST_RUNS / run_name)
            if moved and s is srcs[-1]:
                print("  %-6s %d checkpoints; repointed %s -> %s"
                      % (name, len(srcs), Path(moved[0]).name,
                         Path(moved[1]).name))

    print("\nDone. Next:")
    print("  python plot_training.py")
    print("  python main.py --model llama-ft \"aq nk mandi\"")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--from", dest="src", required=True,
                   help="folder holding the lemmatizer_lora_* directories")
    p.add_argument("--dry-run", action="store_true",
                   help="report what would be imported, copy nothing")
    p.add_argument("--keep-existing", action="store_true",
                   help="do not overwrite adapters already in unsloth/")
    a = p.parse_args()
    main(Path(a.src), a.dry_run, a.keep_existing)
