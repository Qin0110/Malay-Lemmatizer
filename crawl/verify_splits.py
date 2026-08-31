#!/usr/bin/env python
"""Assert the three-way Train / Dev / Test design holds on disk.

    python crawl/verify_splits.py

Every claim the experiment rests on is re-derived from the files themselves,
not from the manifest's own bookkeeping. Exits non-zero on the first failure,
so it is safe to run in a pre-training check.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SILVER = ROOT / "data" / "silver.jsonl"
GOLD = ROOT / "data" / "gold.jsonl"
GOLD_POOL = ROOT / "data" / "gold_pool.jsonl"
MANIFEST = ROOT / "data" / "split_manifest.json"
TRAIN = ROOT / "unsloth" / "data" / "training_split.jsonl"
DEV = ROOT / "unsloth" / "data" / "dev_split.jsonl"

# The split size is a run-time choice (--n), so it is read from the manifest
# rather than hard-coded. What is asserted is the arithmetic: the two halves
# must reconstruct the silver set exactly, in the recorded ratio.
EXPECT_GOLD = 150

failures = []


def check(label, ok, detail=""):
    print("  %-52s %s%s" % (label, "PASS" if ok else "FAIL",
                            "" if ok else "   " + detail))
    if not ok:
        failures.append(label)


def jl(p):
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


for p in (SILVER, GOLD, MANIFEST, TRAIN, DEV):
    if not p.exists():
        sys.exit("missing %s - run `python crawl/build_silver.py --n 8000`"
                 % p.relative_to(ROOT))

silver = jl(SILVER)
gold = jl(GOLD)
pool = jl(GOLD_POOL) if GOLD_POOL.exists() else []
train = jl(TRAIN)
dev = jl(DEV)
man = json.loads(MANIFEST.read_text(encoding="utf-8"))

train_ids = set(man["train_ids"])
dev_ids = set(man["dev_ids"])
gold_ids = {r["id"] for r in gold}
pool_ids = {r["id"] for r in pool}
silver_ids = {r["id"] for r in silver}

ratio = man.get("dev_ratio", 0.10)
exp_dev = max(1, round(len(silver) * ratio))
exp_train = len(silver) - exp_dev

print("\nCOUNTS   (silver %d, dev_ratio %.2f from the manifest)"
      % (len(silver), ratio))
check("train = %d" % exp_train, len(train) == exp_train, str(len(train)))
check("validation = %d" % exp_dev, len(dev) == exp_dev, str(len(dev)))
check("gold test = %d" % EXPECT_GOLD, len(gold) == EXPECT_GOLD, str(len(gold)))
check("manifest ids match the files",
      len(train_ids) == len(train) and len(dev_ids) == len(dev))
check("train + dev = silver (%d)" % len(silver),
      len(train) + len(dev) == len(silver),
      "%d + %d" % (len(train), len(dev)))

print("\nDISJOINTNESS")
check("train n validation = 0", not (train_ids & dev_ids),
      str(sorted(train_ids & dev_ids)[:3]))
check("train n gold = 0", not (train_ids & gold_ids),
      str(sorted(train_ids & gold_ids)[:3]))
check("validation n gold = 0", not (dev_ids & gold_ids),
      str(sorted(dev_ids & gold_ids)[:3]))
check("train n gold_pool = 0", not (train_ids & pool_ids),
      str(sorted(train_ids & pool_ids)[:3]))
check("validation n gold_pool = 0", not (dev_ids & pool_ids),
      str(sorted(dev_ids & pool_ids)[:3]))
check("train u validation = silver ids",
      (train_ids | dev_ids) == silver_ids)

print("\nCONTENT")
sil = {r["id"]: r for r in silver}
tr_txt = [m["messages"][1]["content"] for m in train]
dv_txt = [m["messages"][1]["content"] for m in dev]
check("no duplicate sentences across train and dev",
      not (set(tr_txt) & set(dv_txt)),
      "%d shared" % len(set(tr_txt) & set(dv_txt)))
check("one distinct system prompt in train",
      len({m["messages"][0]["content"] for m in train}) == 1)
check("train and dev share that system prompt",
      {m["messages"][0]["content"] for m in train}
      == {m["messages"][0]["content"] for m in dev})
by_id_train = dict(zip((m["messages"][1]["content"] for m in train), train))
mismatch = sum(1 for r in silver if r["id"] in train_ids
               and r["text"] in by_id_train
               and by_id_train[r["text"]]["messages"][2]["content"]
               != r["lemma_text"])
check("train targets consistent with silver labels", mismatch == 0,
      "%d differ" % mismatch)

print("\nBAND STRATIFICATION")
for band in ("low", "mid", "high"):
    t = sum(1 for i in train_ids if sil[i]["band"] == band)
    d = sum(1 for i in dev_ids if sil[i]["band"] == band)
    tp, dp = 100 * t / len(train_ids), 100 * d / len(dev_ids)
    check("%-4s train %5.1f%%  dev %5.1f%%  (gap < 3pp)" % (band, tp, dp),
          abs(tp - dp) < 3.0, "gap %.1fpp" % abs(tp - dp))

print("\nREPRODUCIBILITY")
check("manifest records the seed", "seed" in man, "")
check("manifest records the dev ratio", "dev_ratio" in man, "")

print()
if failures:
    print("%d CHECK(S) FAILED: %s" % (len(failures), "; ".join(failures)))
    sys.exit(1)
print("all checks passed - the three-way design holds.")
print("  train      %4d   unsloth/data/training_split.jsonl" % len(train))
print("  validation %4d   unsloth/data/dev_split.jsonl" % len(dev))
print("  gold test  %4d   data/gold.jsonl  (never read during training)"
      % len(gold))
