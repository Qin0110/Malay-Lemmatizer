#!/usr/bin/env python
"""
Draw the VALIDATION pool: 150 unused corpus sentences, for hand-labelling.

    python crawl/make_valpool.py              # draw data/val_pool.jsonl
    python crawl/make_valpool.py --n 200
    python crawl/make_valpool.py --dry-run    # stats only, write nothing

Then build the labelling sheet from it:

    python crawl/make_goldsheet.py --pool data/val_pool.jsonl \\
                                   --sheet data/val_sheet.xlsx \\
                                   --out   data/val.jsonl
    ... label the sheet ...
    python crawl/make_goldsheet.py --collect --pool data/val_pool.jsonl \\
                                   --sheet data/val_sheet.xlsx \\
                                   --out   data/val.jsonl

WHY THIS SET HAS TO EXIST
-------------------------
The four systems being compared -- two base models and two fine-tunes -- have
to be ranked on something, and the choice cannot be made on the silver dev
split. Silver labels are the rule pipeline's own output, so a model that
imitates the rules perfectly scores ~100% there BY CONSTRUCTION. Ranking on
that measures similarity to the baseline, not lemmatisation quality.

It also cannot be made on data/gold.jsonl. That is the test set. Selecting a
model on the test set and then reporting the test score is the same mistake as
training on it: the reported number stops being an estimate of anything.

So: a third human-labelled split, drawn from sentences no other split has
touched.

WHY 150, AND WHAT 150 CANNOT DO
-------------------------------
Sized to match data/gold.jsonl exactly, so "the validation ranking held on
test" is a statement about the models rather than about two different sample
sizes.

At 150 sentences (~1,800 tokens, ~600 of them needing a change), a paired
McNemar test at 80% power resolves a gap of roughly:

    3.6 percentage points on all tokens
    6.3 percentage points on the CHANGED subset

That is comfortably enough to separate a base model from a fine-tuned one --
that gap is tens of points. It is NOT enough to rank two fine-tunes that land
within a few points of each other. Resolving a 2-point gap on the CHANGED
subset would need ~1,000 hand-labelled sentences, which is not a realistic
labelling budget, so the honest move when the two fine-tunes come out close is
to report them as tied and break the tie on a pre-declared secondary
criterion. Decide that criterion BEFORE looking at the numbers.

DISJOINTNESS IS ASSERTED, NOT ASSUMED
-------------------------------------
The whole design rests on this set having been seen by nothing else. That is
checked against the files themselves at the bottom of this script, and the
script exits non-zero if it fails.
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus.jsonl"
SILVER = ROOT / "data" / "silver.jsonl"
GOLD = ROOT / "data" / "gold.jsonl"
GOLD_POOL = ROOT / "data" / "gold_pool.jsonl"
VAL_POOL = ROOT / "data" / "val_pool.jsonl"

# Matches data/gold.jsonl: 150 sentences, 50 per difficulty band. Equal rather
# than proportional stratification, because the bands are not equally
# interesting -- `high` is where the rule baseline breaks down and where the
# fine-tune has to earn its keep, so it gets the same weight as `low` rather
# than its 33% share of the corpus.
DEFAULT_N = 150
BANDS = ("low", "mid", "high")

# Distinct from build_corpus.SEED (3407) purely so that a future reader can
# tell the two draws apart in a log. The exclusion set below is what actually
# guarantees no overlap; the seed only makes this draw reproducible.
SEED = 20260903


def jl(path):
    """Every row of a JSONL file."""
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def ids_in(path):
    """Just the ids, without holding the whole file."""
    if not path.exists():
        return set()
    out = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.add(json.loads(line)["id"])
    return out


def main(n, seed, dry_run):
    if not CORPUS.exists():
        sys.exit("No %s. Run crawl/build_corpus.py first."
                 % CORPUS.relative_to(ROOT))

    # Everything any other split has ever seen. GOLD_POOL is included
    # separately from GOLD because a pool row that was drawn but never
    # labelled is still spent: it may have been read by an annotator.
    spent = ids_in(SILVER) | ids_in(GOLD) | ids_in(GOLD_POOL)

    corpus = jl(CORPUS)
    available = [r for r in corpus if r["id"] not in spent]

    print("corpus            %6d" % len(corpus))
    print("already spent     %6d   (silver + gold + gold_pool)" % len(spent))
    print("available         %6d" % len(available))

    by_band = {b: [] for b in BANDS}
    for row in available:
        by_band.setdefault(row.get("band"), []).append(row)

    print("\navailable by band")
    for b in BANDS:
        print("  %-5s %6d" % (b, len(by_band[b])))

    per_band = n // len(BANDS)
    short = [b for b in BANDS if len(by_band[b]) < per_band]
    if short:
        print("\nWARNING: %s cannot supply %d sentences; the draw will be "
              "topped up from the largest band and the bands will NOT be "
              "balanced. Say so in the report if this fires."
              % (", ".join(short), per_band))

    rng = random.Random(seed)
    pool = []
    for b in BANDS:
        pool += rng.sample(by_band[b], min(per_band, len(by_band[b])))

    if len(pool) < n:
        chosen = {r["id"] for r in pool}
        rest = [r for r in available if r["id"] not in chosen]
        pool += rng.sample(rest, min(n - len(pool), len(rest)))

    # Shuffle so the sheet does not present all the easy sentences first --
    # annotator fatigue would then land entirely on the `high` band, which is
    # the band whose labels matter most.
    rng.shuffle(pool)

    # ---- assertions: the design, re-derived from the files ----------------
    pool_ids = {r["id"] for r in pool}
    ok = True

    def check(label, condition, detail=""):
        nonlocal ok
        print("  %-46s %s%s" % (label, "PASS" if condition else "FAIL",
                                "" if condition else "   " + detail))
        ok &= condition

    print("\nDISJOINTNESS")
    check("val n silver = 0", not (pool_ids & ids_in(SILVER)))
    check("val n gold = 0", not (pool_ids & ids_in(GOLD)))
    check("val n gold_pool = 0", not (pool_ids & ids_in(GOLD_POOL)))
    check("no duplicate ids within val", len(pool_ids) == len(pool),
          "%d rows, %d distinct" % (len(pool), len(pool_ids)))
    check("drew the full %d" % n, len(pool) == n, str(len(pool)))

    bands = Counter(r["band"] for r in pool)
    # A pool row is not tokenised yet -- that happens when the sheet is built.
    # n_words is the corpus's own whitespace count, close enough to size the
    # labelling job and to sanity-check the power calculation in the docstring.
    n_tok = sum(r["n_words"] for r in pool)
    print("\nDRAWN     %d sentences, ~%d tokens (%.1f per sentence)"
          % (len(pool), n_tok, n_tok / max(len(pool), 1)))
    for b in BANDS:
        print("  %-5s %4d" % (b, bands[b]))

    if not ok:
        sys.exit("\nDisjointness failed - not written.")
    if dry_run:
        print("\n--dry-run: nothing written.")
        return

    with VAL_POOL.open("w", encoding="utf-8") as fh:
        for row in pool:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("\nwrote %s" % VAL_POOL.relative_to(ROOT))
    print("\nNext:\n  python crawl/make_goldsheet.py --pool data/val_pool.jsonl "
          "--sheet data/val_sheet.xlsx --out data/val.jsonl")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--n", type=int, default=DEFAULT_N,
                   help="how many sentences (default %d)" % DEFAULT_N)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--dry-run", action="store_true",
                   help="report the draw without writing it")
    a = p.parse_args()
    main(a.n, a.seed, a.dry_run)
