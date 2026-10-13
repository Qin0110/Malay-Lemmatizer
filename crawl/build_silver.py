#!/usr/bin/env python
"""Label the corpus with the rules and split it into train / dev.

    python crawl/build_silver.py            label everything, write silver + split
    python crawl/build_silver.py --check    is silver.jsonl still what the rules say?
    python crawl/build_silver.py --dry-run

Sentences in the gold and val sets (and their pools) are never used. A
sentence keeps its train/dev side from the previous split; new sentences are
split 90/10 within each difficulty band.
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rules                                                      # noqa: E402

DATA = ROOT / "data"
CORPUS = DATA / "corpus.jsonl"
SILVER = DATA / "silver.jsonl"
MANIFEST = DATA / "split_manifest.json"
DEV_RATIO = 0.10
SEED = 3407


def jl(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def held_out_ids():
    """Ids in any labelled set or labelling pool - never used for training."""
    paths = [DATA / "gold.jsonl", DATA / "val.jsonl"] + sorted(DATA.glob("*_pool.jsonl"))
    return {r["id"] for path in paths for r in jl(path)}


def label_row(row):
    labels = rules.label(row["text"])
    return {
        "id": row["id"],
        "text": row["text"],
        "tokens": [e["t"] for e in labels],
        "lemmas": labels,
        "lemma_text": rules.lemma_text(labels),
        "band": row.get("band"),
    }


def split(rows, old):
    """Keep each sentence's previous side; split new ones by band."""
    train, dev, new = [], [], defaultdict(list)
    for r in rows:
        side = old.get(r["id"])
        if side == "train":
            train.append(r)
        elif side == "dev":
            dev.append(r)
        else:
            new[r["band"]].append(r)

    rng = random.Random(SEED)
    for band in sorted(new):
        group = sorted(new[band], key=lambda r: r["id"])
        rng.shuffle(group)
        n_dev = round(len(group) * DEV_RATIO)
        dev += group[:n_dev]
        train += group[n_dev:]
    return train, dev


def check():
    silver = jl(SILVER)
    if not silver:
        sys.exit("No data/silver.jsonl yet.")
    print("loading MALINDO...")
    stale, examples = 0, Counter()
    for row in silver:
        fresh = rules.label(row["text"])
        for old, new in zip(row["lemmas"], fresh):
            if (old["t"], old["l"], old["src"]) != (new["t"], new["l"], new["src"]):
                stale += 1
                examples[(new["t"], old["l"], new["l"])] += 1
        if len(fresh) != len(row["lemmas"]):
            stale += 1
    if not stale:
        print("silver.jsonl matches the current rules exactly.")
        return 0
    print("%d tokens in silver.jsonl no longer match the rules, e.g.:" % stale)
    for (t, old, new), n in examples.most_common(15):
        print("  %-14s %-14s -> %-14s x%d" % (t, old, new, n))
    print("\nRun python crawl/build_silver.py, then python crawl/build_sft.py")
    return 1


def main(limit, dry_run):
    corpus = jl(CORPUS)
    if not corpus:
        sys.exit("No data/corpus.jsonl.")
    skip = held_out_ids()
    pool = [r for r in corpus if r["id"] not in skip]
    if limit:
        pool = pool[:limit]
    print("corpus %d, held out %d, labelling %d" % (len(corpus), len(skip), len(pool)))

    old = {}
    if MANIFEST.exists():
        man = json.loads(MANIFEST.read_text(encoding="utf-8"))
        old = {i: "train" for i in man["train_ids"]}
        old.update({i: "dev" for i in man["dev_ids"]})

    print("loading MALINDO...")
    rows = [label_row(r) for r in pool]
    rows = [r for r in rows if r["lemmas"]]
    train, dev = split(rows, old)

    t_ids, d_ids = {r["id"] for r in train}, {r["id"] for r in dev}
    assert not (t_ids & d_ids), "train and dev overlap"
    assert not ((t_ids | d_ids) & skip), "a held-out sentence reached the training data"

    src = Counter(e["src"] for r in rows for e in r["lemmas"])
    total = sum(src.values())
    print("\n%d sentences, %d tokens   train %d / dev %d"
          % (len(rows), total, len(train), len(dev)))
    for name, n in src.most_common():
        print("  %-11s %7d %5.1f%%" % (name, n, 100 * n / total))

    if dry_run:
        print("\n--dry-run: nothing written.")
        return

    with SILVER.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    MANIFEST.write_text(json.dumps({
        "seed": SEED,
        "dev_ratio": DEV_RATIO,
        "counts": {"train": len(train), "dev": len(dev), "held_out": len(skip)},
        "band_mix": {
            "train": dict(Counter(r["band"] for r in train)),
            "dev": dict(Counter(r["band"] for r in dev)),
        },
        "train_ids": sorted(t_ids),
        "dev_ids": sorted(d_ids),
    }, indent=1), encoding="utf-8")
    print("\nwrote data/silver.jsonl and data/split_manifest.json")
    print("Next: python crawl/build_sft.py")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="Rule-label the corpus for training.")
    p.add_argument("--check", action="store_true",
                   help="compare silver.jsonl with what the rules say now")
    p.add_argument("--limit", type=int, default=0,
                   help="only label the first N sentences (default: all)")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    sys.exit(check()) if a.check else main(a.limit, a.dry_run)
