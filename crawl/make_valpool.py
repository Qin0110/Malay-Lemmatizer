#!/usr/bin/env python
"""Draw a new pool of sentences to label by hand.

    python crawl/make_valpool.py --out data/test2_pool.jsonl
    python crawl/make_valpool.py --out data/test2_pool.jsonl --n 200 --dry-run

The pool is balanced across the difficulty bands and never overlaps another
labelled set. Once drawn it is held out of training; rebuild the training
data afterwards with crawl/build_silver.py and crawl/build_sft.py.
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "crawl"))

import build_silver                                               # noqa: E402

BANDS = ("low", "mid", "high")


def main(out, n, seed, dry_run):
    if out.exists():
        sys.exit("%s already exists. It may already be labelled - pick a new "
                 "--out name." % out.relative_to(ROOT))

    corpus = build_silver.jl(build_silver.CORPUS)
    spent = build_silver.held_out_ids()
    available = [r for r in corpus if r["id"] not in spent]
    print("corpus %d, already held out %d, available %d"
          % (len(corpus), len(spent), len(available)))

    by_band = {b: [] for b in BANDS}
    for row in available:
        by_band.setdefault(row.get("band"), []).append(row)

    rng = random.Random(seed)
    pool = []
    for b in BANDS:
        pool += rng.sample(by_band[b], min(n // len(BANDS), len(by_band[b])))
    if len(pool) < n:
        chosen = {r["id"] for r in pool}
        rest = [r for r in available if r["id"] not in chosen]
        pool += rng.sample(rest, min(n - len(pool), len(rest)))
    rng.shuffle(pool)

    print("drew %d sentences: %s" % (len(pool), dict(Counter(r["band"] for r in pool))))
    if dry_run:
        print("--dry-run: nothing written.")
        return

    with out.open("w", encoding="utf-8") as fh:
        for row in pool:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    sheet = out.with_name(out.stem.replace("_pool", "") + "_sheet.xlsx")
    final = out.with_name(out.stem.replace("_pool", "") + ".jsonl")
    print("wrote %s\n" % out.relative_to(ROOT))
    print("Next:")
    print("  python crawl/make_goldsheet.py --pool %s --sheet %s --out %s"
          % (out.relative_to(ROOT).as_posix(), sheet.relative_to(ROOT).as_posix(),
             final.relative_to(ROOT).as_posix()))
    print("  python crawl/build_silver.py && python crawl/build_sft.py")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Draw a labelling pool.")
    p.add_argument("--out", type=Path, required=True,
                   help="new pool file, e.g. data/test2_pool.jsonl")
    p.add_argument("--n", type=int, default=150)
    p.add_argument("--seed", type=int, default=20260903)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    out = a.out if a.out.is_absolute() else (ROOT / a.out)
    if not out.name.endswith("_pool.jsonl"):
        p.error("--out must end in _pool.jsonl so it is held out of training")
    main(out.resolve(), a.n, a.seed, a.dry_run)
