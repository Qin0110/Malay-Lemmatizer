#!/usr/bin/env python
"""
Rebuild the fine-tuning targets from the silver annotations.

    python crawl/build_sft.py                 # write train + dev ChatML
    python crawl/build_sft.py --dry-run       # stats and a sample, write nothing
    python crawl/build_sft.py --show 2        # print 2 rendered examples

WHAT THIS REPLACES
------------------
unsloth/data/training_split.jsonl was written for the old skill, whose target
was a bare lemma sentence:

    aku siap ada playlist sembang seram kakak jem sebab memang minat gila

The current skill asks for a derivation -- one line per token, each carrying
the rule that produced its lemma, then the sentence. Training on the old file
would produce a model that ignores the reasoning format entirely, which is the
only thing that makes its answers auditable.

WHERE THE TARGETS COME FROM
---------------------------
Not from the old ChatML, which has no per-token information left in it. From
data/silver.jsonl, which carries `{"t": token, "l": lemma, "src": rule}` for
every token, through skill.render_target(). So the derivations are RENDERED
from provenance the corpus already recorded, never invented -- if a target says
IMBUHAN, it is because the pipeline actually applied an affix rule to that
token.

WHICH ROWS GO WHERE
-------------------
data/split_manifest.json decides, not a fresh shuffle. Re-splitting here would
silently move sentences between train and dev between runs, so a model trained
today and one trained tomorrow would not be comparable, and a sentence could
drift into the split it was meant to be held out from.

The gold test set never appears. That is asserted, not assumed.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import skill                                                  # noqa: E402

SILVER = ROOT / "data" / "silver.jsonl"
GOLD = ROOT / "data" / "gold.jsonl"
MANIFEST = ROOT / "data" / "split_manifest.json"
OUT_DIR = ROOT / "unsloth" / "data"
TRAIN_OUT = OUT_DIR / "train_reasoning.jsonl"
DEV_OUT = OUT_DIR / "dev_reasoning.jsonl"


def jl(path):
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main(dry_run, show):
    for path in (SILVER, MANIFEST):
        if not path.exists():
            sys.exit("missing %s" % path.relative_to(ROOT))

    silver = {r["id"]: r for r in jl(SILVER)}
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    train_ids = [i for i in man["train_ids"] if i in silver]
    dev_ids = [i for i in man["dev_ids"] if i in silver]
    gold_ids = {r["id"] for r in jl(GOLD)} if GOLD.exists() else set()

    print("silver %d | manifest: train %d, dev %d"
          % (len(silver), len(man["train_ids"]), len(man["dev_ids"])))

    # ---- the assertions the experiment rests on --------------------------
    ok = True

    def check(label, condition, detail=""):
        nonlocal ok
        print("  %-44s %s%s" % (label, "PASS" if condition else "FAIL",
                                "" if condition else "   " + detail))
        ok &= condition

    print("\nSPLIT INTEGRITY")
    check("train n dev = 0", not (set(train_ids) & set(dev_ids)))
    check("train n gold = 0", not (set(train_ids) & gold_ids))
    check("dev n gold = 0", not (set(dev_ids) & gold_ids))
    check("every manifest id resolves in silver",
          len(train_ids) == len(man["train_ids"])
          and len(dev_ids) == len(man["dev_ids"]),
          "%d/%d train, %d/%d dev" % (len(train_ids), len(man["train_ids"]),
                                      len(dev_ids), len(man["dev_ids"])))
    if not ok:
        sys.exit("\nSplit integrity failed - nothing written.")

    # ---- render ----------------------------------------------------------
    tags = Counter()
    splits = []
    for name, ids, out in (("train", train_ids, TRAIN_OUT),
                           ("dev", dev_ids, DEV_OUT)):
        rows = []
        for sid in ids:
            row = silver[sid]
            rows.append(skill.to_chatml(row))
            for e in row["lemmas"]:
                tag, _unc = skill.rule_for(e["src"], e["t"], e["l"])
                tags[tag] += 1
        splits.append((name, rows, out))

    print("\nRULE TAGS ACROSS THE TARGETS")
    total = sum(tags.values())
    for tag, n in tags.most_common():
        print("  %-9s %6d  %4.1f%%" % (tag, n, 100 * n / total))
    missing = [t for t in skill.RULES if not tags[t]]
    if missing:
        print("\n  WARNING: %s appear in the system prompt but never in the "
              "targets. A tag with no training examples is one the model will "
              "apply at random." % ", ".join(missing))

    if show:
        for name, rows, _ in splits:
            for row in rows[:show]:
                print("\n" + "=" * 68)
                print("[%s] %s" % (name, row["messages"][1]["content"]))
                print("-" * 68)
                print(row["messages"][2]["content"])
            break

    if dry_run:
        print("\n--dry-run: nothing written.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows, out in splits:
        with out.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print("\n%-5s %5d rows -> %s" % (name, len(rows), out.relative_to(ROOT)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--show", type=int, default=0,
                   help="print N rendered examples")
    a = p.parse_args()
    main(a.dry_run, a.show)
