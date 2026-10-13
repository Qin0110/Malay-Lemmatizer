#!/usr/bin/env python
"""Check that train, dev, val and gold are kept apart and in sync.

    python crawl/verify_splits.py

Exits with 1 if any check fails, so it can run before training.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "crawl"))

import build_silver                                               # noqa: E402
import skill                                                      # noqa: E402

DATA = ROOT / "data"
TRAIN = DATA / "sft" / "train.jsonl"
DEV = DATA / "sft" / "dev.jsonl"

failures = []


def check(label, ok, detail=""):
    print("  %-50s %s%s" % (label, "PASS" if ok else "FAIL",
                            "" if ok else "   " + detail))
    if not ok:
        failures.append(label)


def ids(path):
    return {r["id"] for r in build_silver.jl(path)}


def main():
    for path in (build_silver.SILVER, build_silver.MANIFEST, TRAIN, DEV):
        if not path.exists():
            sys.exit("missing %s - run crawl/build_silver.py and crawl/build_sft.py"
                     % path.relative_to(ROOT))

    silver = {r["id"]: r for r in build_silver.jl(build_silver.SILVER)}
    man = json.loads(build_silver.MANIFEST.read_text(encoding="utf-8"))
    train_ids, dev_ids = set(man["train_ids"]), set(man["dev_ids"])
    train, dev = build_silver.jl(TRAIN), build_silver.jl(DEV)
    gold, val = ids(DATA / "gold.jsonl"), ids(DATA / "val.jsonl")
    held = build_silver.held_out_ids()

    print("\nSIZES")
    check("train file matches the manifest (%d)" % len(train_ids), len(train) == len(train_ids),
          str(len(train)))
    check("dev file matches the manifest (%d)" % len(dev_ids), len(dev) == len(dev_ids),
          str(len(dev)))
    check("every manifest id is in silver", (train_ids | dev_ids) <= set(silver))

    print("\nNO OVERLAP")
    check("train and dev", not (train_ids & dev_ids))
    check("train/dev and gold (%d)" % len(gold), not ((train_ids | dev_ids) & gold))
    check("train/dev and val (%d)" % len(val), not ((train_ids | dev_ids) & val))
    check("train/dev and every labelling pool", not ((train_ids | dev_ids) & held))
    check("gold and val", not (gold & val))
    texts = lambda rows: {r["messages"][1]["content"] for r in rows}
    check("no sentence in both train and dev", not (texts(train) & texts(dev)))

    print("\nTARGETS")
    check("one system prompt, the current one",
          {r["messages"][0]["content"] for r in train + dev} == {skill.SYSTEM_PROMPT})
    by_text = {silver[i]["text"]: silver[i] for i in train_ids | dev_ids if i in silver}
    bad = sum(1 for r in train + dev
              if skill.render_target(by_text[r["messages"][1]["content"]])
              != r["messages"][2]["content"])
    check("every target matches its silver labels", bad == 0, "%d differ" % bad)

    print()
    if failures:
        print("%d check(s) failed." % len(failures))
        sys.exit(1)
    print("All checks passed: train %d, dev %d, val %d, gold %d."
          % (len(train), len(dev), len(val), len(gold)))


if __name__ == "__main__":
    main()
