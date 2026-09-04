#!/usr/bin/env python
"""Re-derive the filtering cascade reported in Section 3.4, writing nothing.

    python crawl/verify_filter.py

WHY THIS EXISTS SEPARATELY FROM build_corpus.py
-----------------------------------------------
build_corpus.py prints the same cascade, but it also REWRITES
data/corpus.jsonl, and that file must not be regenerated. Sentence ids are
assigned by position as sentences are kept:

    "id": "yt_%05d" % len(kept)

so admitting even one extra sentence shifts every id after it. The corpus on
disk was produced by an earlier run of the same pipeline, and re-running the
current filter keeps 193 more sentences -- which would re-point 98.9% of ids at
different text and detach 294 of the 300 hand-verified gold and validation
labels from the sentences they describe.

So the numbers have to be reproducible WITHOUT touching the corpus. This script
replays the same five filters over the same raw comments, in the same order,
using build_corpus's own functions, and only prints.

WHAT IT PROVES
--------------
  1. the cascade in the report is reproducible to the sentence
  2. the corpus in use is a strict SUBSET of what those filters admit --
     nothing in the experiment fails the published criteria
  3. how far the corpus is from a rebuild, in ids
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "crawl"))

import build_corpus as bc                                       # noqa: E402

RAW = ROOT / "data" / "raw" / "youtube_raw.jsonl"
CORPUS = ROOT / "data" / "corpus.jsonl"
GOLD = ROOT / "data" / "gold.jsonl"
VAL = ROOT / "data" / "val.jsonl"


def jl(path):
    with path.open(encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def replay():
    """The five filters, in build_corpus's order. Returns (kept, drop counts).

    Deliberately a copy of the loop rather than a call into build(), because
    build() writes. If build_corpus's filtering changes, this will disagree
    with it -- which is a result worth seeing, not a bug to paper over.
    """
    raw = jl(RAW)
    kept, seen, drop = [], set(), Counter()

    for comment in raw:
        for sent in bc.sentences(bc.clean(comment["text"])):
            n_words = len(sent.split())
            if not (bc.MIN_WORDS <= n_words <= bc.MAX_WORDS):
                drop["length"] += 1
                continue

            key = " ".join(re.sub(r"[^a-z0-9 ]", "", sent.lower()).split())
            if not key or key in seen:
                drop["duplicate"] += 1
                continue
            seen.add(key)

            if bc.latin_ratio(sent) < 0.8:
                drop["script"] += 1
                continue

            f, malay_hits, _n_tok, oov = bc.features(sent, True)
            hits = sum(f.values())
            if malay_hits < (1 if n_words <= 6 else bc.MIN_MALAY_HITS):
                drop["not_malay"] += 1
                continue

            if not bc.keeps(hits, malay_hits, n_words, bc.MIN_ROJAK_HITS, oov):
                drop["not_rojak"] += 1
                continue

            kept.append(sent)

    return raw, kept, drop


LABELS = [
    ("length",     "Length",         "retain 3-30 words"),
    ("duplicate",  "Duplicate",      "drop repeated normalised sentences"),
    ("script",     "Script",         ">= 80% Latin characters"),
    ("not_malay",  "Malay evidence", "sufficient Malay-language evidence"),
    ("not_rojak",  "Rojak evidence", "sufficient rojak characteristics"),
]


def main():
    raw, kept, drop = replay()
    corpus = jl(CORPUS)

    print("\n%d raw comments" % len(raw))
    print("%d candidate sentences after segmentation\n"
          % (len(kept) + sum(drop.values())))

    print("%-16s %-38s %9s" % ("filter", "criterion", "removed"))
    print("-" * 66)
    for key, label, criterion in LABELS:
        print("%-16s %-38s %9d" % (label, criterion, drop[key]))
    print("-" * 66)
    print("%-55s %9d" % ("total removed", sum(drop.values())))
    print("%-55s %9d" % ("retained by the filter as specified", len(kept)))
    print("%-55s %9d" % ("corpus in use (data/corpus.jsonl)", len(corpus)))

    # ---- the subset claim, which is the one that matters -----------------
    rerun = set(kept)
    disk = {r["text"] for r in corpus}
    print("\ncomparison by sentence text, not by id")
    print("   in both                                        %6d" % len(rerun & disk))
    print("   only in the re-run                             %6d" % len(rerun - disk))
    print("   only in the corpus -- would FAIL the criteria   %6d" % len(disk - rerun))
    if not (disk - rerun):
        print("   => every sentence in the corpus passes the published criteria")

    # ---- what a rebuild would cost ---------------------------------------
    rebuilt = {"yt_%05d" % i: s for i, s in enumerate(kept)}
    on_disk = {r["id"]: r["text"] for r in corpus}
    shared = set(rebuilt) & set(on_disk)
    moved = sum(1 for i in shared if rebuilt[i] != on_disk[i])
    print("\nif the corpus were regenerated, ids being positional:")
    print("   ids pointing at different text   %6d of %d  (%.1f%%)"
          % (moved, len(shared), 100 * moved / max(len(shared), 1)))
    for name, path in (("gold", GOLD), ("val", VAL)):
        if path.exists():
            rows = jl(path)
            bad = sum(1 for r in rows if rebuilt.get(r["id"]) != r["text"])
            print("   %-4s labels detached from their sentence  %4d of %d"
                  % (name, bad, len(rows)))
    print("\nnothing was written.\n")


if __name__ == "__main__":
    main()
