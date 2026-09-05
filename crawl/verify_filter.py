#!/usr/bin/env python


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
