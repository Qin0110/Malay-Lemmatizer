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
DEFAULT_N = 150
BANDS = ("low", "mid", "high")
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

    rng.shuffle(pool)

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
