#!/usr/bin/env python
"""Render the labelled data as chat examples for fine-tuning.

    python crawl/build_sft.py              -> data/sft/train.jsonl, data/sft/dev.jsonl
    python crawl/build_sft.py --show 2     print a couple of rendered examples

Stops if data/silver.jsonl is out of date with the rules, so the model is
never trained on labels the app would contradict.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "crawl"))

import build_silver                                               # noqa: E402
import skill                                                      # noqa: E402

SFT_DIR = ROOT / "data" / "sft"
TRAIN_OUT = SFT_DIR / "train.jsonl"
DEV_OUT = SFT_DIR / "dev.jsonl"


def main(dry_run, show, skip_check):
    silver = {r["id"]: r for r in build_silver.jl(build_silver.SILVER)}
    if not silver or not build_silver.MANIFEST.exists():
        sys.exit("Run python crawl/build_silver.py first.")
    if not skip_check and build_silver.check() != 0:
        sys.exit("\nNothing written.")

    man = json.loads(build_silver.MANIFEST.read_text(encoding="utf-8"))
    held = build_silver.held_out_ids()
    train_ids = [i for i in man["train_ids"] if i in silver]
    dev_ids = [i for i in man["dev_ids"] if i in silver]
    if set(train_ids) & set(dev_ids) or (set(train_ids) | set(dev_ids)) & held:
        sys.exit("The split overlaps the dev, gold or val sets - nothing written.")

    tags = Counter()
    splits = []
    for name, ids, out in (("train", train_ids, TRAIN_OUT), ("dev", dev_ids, DEV_OUT)):
        rows = []
        for sid in ids:
            rows.append(skill.to_chatml(silver[sid]))
            for e in silver[sid]["lemmas"]:
                tags[skill.rule_for(e["src"])[0]] += 1
        splits.append((name, rows, out))

    total = sum(tags.values())
    print("\nrule tags in the targets")
    for tag, n in tags.most_common():
        print("  %-9s %7d %5.1f%%" % (tag, n, 100 * n / total))

    for _, rows, _ in splits[:1]:
        for row in rows[:show]:
            print("\n" + "=" * 68)
            print(row["messages"][1]["content"])
            print("-" * 68)
            print(row["messages"][2]["content"])

    if dry_run:
        print("\n--dry-run: nothing written.")
        return

    SFT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows, out in splits:
        with out.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print("%-5s %5d examples -> %s" % (name, len(rows), out.relative_to(ROOT)))
    print("\nNext: python train.py --model llama")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="Render the training examples.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--show", type=int, default=0, help="print N examples")
    p.add_argument("--skip-check", action="store_true",
                   help="do not compare silver.jsonl with the rules first")
    a = p.parse_args()
    main(a.dry_run, a.show, a.skip_check)
