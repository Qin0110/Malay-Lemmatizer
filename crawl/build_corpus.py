#!/usr/bin/env python
"""Add rojak sentences from data/raw/*.jsonl to data/corpus.jsonl.

    python crawl/build_corpus.py               add whatever is new
    python crawl/build_corpus.py --dry-run     report only
    python crawl/build_corpus.py --validate    also write a hand-check sample

Existing rows and their ids are never changed - the gold and val sets refer
to them by id. New sentences get the next free id for their source
(yt_, tw_, th_, ...).
"""

import argparse
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lexicon import PARTICLES, SLANG                               # noqa: E402
from preprocess import clean, latin_ratio, sentences               # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
CORPUS_FILE = ROOT / "data" / "corpus.jsonl"
MALINDO_TSV = ROOT / "malindo_morph" / "malindo_dic_2024.tsv"
CHECK_FILE = ROOT / "data" / "filter_check.tsv"
CHECK_XLSX = ROOT / "data" / "filter_check.xlsx"

MIN_WORDS, MAX_WORDS = 3, 30
MIN_MALAY_HITS = 2
MIN_ROJAK_HITS = 2
SEED = 3407

PREFIX = {
    "youtube": "yt", "twitter": "tw", "x": "tw", "threads": "th",
    "reddit": "rd", "facebook": "fb", "instagram": "ig", "tiktok": "tt",
    "forum": "fr",
}

MALAY_FUNCTION = {
    "yang", "dan", "di", "ke", "dari", "untuk", "dengan", "ini", "itu",
    "tidak", "ada", "saya", "dia", "kita", "kami", "mereka", "sudah", "akan",
    "juga", "pada", "adalah", "boleh", "nak", "tak", "ni", "tu", "aku", "kau",
    "apa", "kenapa", "macam", "sebab", "kalau", "bila", "semua", "orang",
    "buat", "jangan", "lagi", "sangat", "memang", "bukan", "kena", "dalam",
}

ENGLISH_MARKERS = {
    "the", "is", "are", "you", "your", "and", "but", "with", "have", "has",
    "this", "that", "for", "not", "can", "will", "just", "like", "get", "all",
    "what", "why", "how", "very", "really", "good", "bad", "best", "so",
    "people", "make", "want", "need", "know", "think", "time", "much",
}

WORD_RE = re.compile(r"[a-z]+\d|[a-z]+(?:-[a-z]+)*|\d+")
X_NEGATION_RE = re.compile(r"\bx[a-z]{2,}\b")
REDUP_DIGIT_RE = re.compile(r"\b[a-z]{3,}2\b")
ELONGATION_RE = re.compile(r"([a-z])\1{2,}")
NO_VOWEL_RE = re.compile(r"\b(?![0-9])[bcdfghjklmnpqrstvwxyz]{3,}\b")
E_FOR_A_RE = re.compile(
    r"\b(ape|mane|kenape|sape|bile|camne|mcmne|ade|jadinye|kite|nape|"
    r"cane|gile|die|dye|kerje|keje|sume|bende|kejap)\b")

_SURFACE = None


def malindo_surfaces():
    global _SURFACE
    if _SURFACE is None:
        import pandas as pd
        df = pd.read_csv(MALINDO_TSV, sep="\t", usecols=[2], names=["surface"],
                         header=None, encoding="utf-8", dtype=str)
        _SURFACE = set(df["surface"].dropna().str.lower())
    return _SURFACE


def features(sentence, use_malindo=True):
    low = sentence.lower()
    words = WORD_RE.findall(low)
    wordset = set(words)

    f = {
        "slang": sum(1 for w in words if w in SLANG),
        "particle": sum(1 for w in words if w in PARTICLES),
        "x_neg": len(X_NEGATION_RE.findall(low)),
        "redup": len(REDUP_DIGIT_RE.findall(low)),
        "elong": len(ELONGATION_RE.findall(low)),
        "no_vowel": len(NO_VOWEL_RE.findall(low)),
        "e_for_a": len(E_FOR_A_RE.findall(low)),
    }
    english_hits = len(wordset & ENGLISH_MARKERS)

    # elongation, reduplication and missing vowels are left out on purpose:
    # they would let "soooo goooood" through as Malay
    malay_hits = (len(wordset & MALAY_FUNCTION)
                  + f["slang"] + f["particle"] + f["e_for_a"] + f["x_neg"])
    f["codeswitch"] = english_hits if (malay_hits >= 2 and english_hits) else 0

    oov = 0
    if use_malindo:
        surfaces = malindo_surfaces()
        oov = sum(1 for w in words if w not in surfaces)

    return f, malay_hits, len(words), oov


def difficulty(hits, oov):
    return oov * 2 + hits


def band(hits, oov=0):
    score = difficulty(hits, oov)
    if score >= 10:
        return "high"
    if score >= 6:
        return "mid"
    return "low"


def keeps(hits, malay_hits, n_words, min_hits, oov=0):
    if oov >= 2 or hits >= 4:
        return True
    if n_words <= 6 and oov >= 1 and hits >= max(1, min_hits - 1):
        return True
    return False


def dedup_key(sentence):
    return " ".join(re.sub(r"[^a-z0-9 ]", "", sentence.lower()).split())


def judge(sent, min_hits=MIN_ROJAK_HITS, use_malindo=True):
    """(verdict, row) for one sentence. verdict is "keep" or the drop reason."""
    n_words = len(sent.split())
    if not (MIN_WORDS <= n_words <= MAX_WORDS):
        return "length", None
    if latin_ratio(sent) < 0.8:
        return "script", None

    f, malay_hits, n_tokens, oov = features(sent, use_malindo)
    hits = sum(f.values())
    if malay_hits < (1 if n_words <= 6 else MIN_MALAY_HITS):
        return "not_malay", None

    row = {
        "text": sent,
        "n_words": n_words,
        "rojak_hits": hits,
        "oov": oov,
        "difficulty": difficulty(hits, oov),
        "rojak_density": round(hits / max(n_tokens, 1), 3),
        "band": band(hits, oov),
        "features": f,
    }
    if not keeps(hits, malay_hits, n_words, min_hits, oov):
        return "not_rojak", row
    return "keep", row


def read_corpus():
    if not CORPUS_FILE.exists():
        return []
    with CORPUS_FILE.open(encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def raw_posts():
    """(source, text) for every post under data/raw/."""
    for path in sorted(RAW_DIR.glob("*.jsonl")):
        default = "youtube" if path.name.startswith("youtube") else path.stem
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    post = json.loads(line)
                    yield post.get("source", default), post.get("text", "")


def build(min_hits, dry_run, validate, use_malindo=True):
    if not RAW_DIR.exists() or not any(RAW_DIR.glob("*.jsonl")):
        sys.exit("Nothing in data/raw/. Collect posts first:\n"
                 "    python crawl/youtube_crawl.py --videos 25\n"
                 "    python crawl/import_posts.py <export file> --source twitter")

    corpus = read_corpus()
    seen = {dedup_key(r["text"]) for r in corpus}
    next_id = defaultdict(int)
    for r in corpus:
        prefix, _, num = r["id"].rpartition("_")
        if num.isdigit():
            next_id[prefix] = max(next_id[prefix], int(num) + 1)

    added, rejected, drop, posts = [], [], Counter(), 0
    for source, text in raw_posts():
        posts += 1
        for sent in sentences(clean(text)):
            key = dedup_key(sent)
            if not key or key in seen:
                drop["duplicate"] += 1
                continue
            seen.add(key)
            verdict, row = judge(sent, min_hits, use_malindo)
            if verdict != "keep":
                drop[verdict] += 1
                if row is not None and validate:
                    rejected.append(row)
                continue
            prefix = PREFIX.get(source.lower(), "sm")
            row = dict({"id": "%s_%05d" % (prefix, next_id[prefix]),
                        "source": source.lower()}, **row)
            next_id[prefix] += 1
            added.append(row)

    report(posts, added, drop, len(corpus))
    if dry_run:
        print("\n--dry-run: nothing written.")
        return
    if added:
        with CORPUS_FILE.open("a", encoding="utf-8") as fh:
            for row in added:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print("\nappended %d sentences -> data/corpus.jsonl" % len(added))
        print("Next: python crawl/build_silver.py")
    if validate:
        write_check(added, rejected)


def report(posts, added, drop, existing):
    print("%d posts read, %d sentences already in the corpus\n" % (posts, existing))
    print("%-14s %7s" % ("dropped", "n"))
    print("-" * 22)
    for reason in ("duplicate", "length", "script", "not_malay", "not_rojak"):
        print("%-14s %7d" % (reason, drop[reason]))
    print("-" * 22)
    print("%-14s %7d" % ("new rojak", len(added)))

    if added:
        print("\nby source:", dict(Counter(r["source"] for r in added)))
        print("by band:  ", dict(Counter(r["band"] for r in added)))
        print("\nsamples:")
        for row in added[:5]:
            print("  [%s] %s" % (row["band"], row["text"][:80]))


def write_check(kept, rejected):
    if CHECK_XLSX.exists():
        archive = CHECK_XLSX.with_name(
            "filter_check_%s.xlsx" % time.strftime("%Y%m%d_%H%M%S"))
        CHECK_XLSX.rename(archive)
        print("previous sheet kept as %s" % archive.name)

    rng = random.Random(SEED)
    sample = ([("KEPT", r) for r in rng.sample(kept, min(100, len(kept)))]
              + [("REJECTED", r) for r in rng.sample(rejected, min(50, len(rejected)))])
    rng.shuffle(sample)
    if not sample:
        print("nothing to sample")
        return

    with CHECK_FILE.open("w", encoding="utf-8") as f:
        f.write("filter_said\tis_really_rojak_Y_or_N\thits\ttext\n")
        for verdict, row in sample:
            f.write("%s\t\t%d\t%s\n" % (verdict, row["rojak_hits"], row["text"]))

    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font
        from openpyxl.worksheet.datavalidation import DataValidation
    except ImportError:
        print("wrote %d rows -> data/filter_check.tsv" % len(sample))
        return

    wb = Workbook()
    ws = wb.active
    ws.title = "filter check"
    ws.append(["filter_said", "hits", "is_really_rojak_Y_or_N", "text"])
    for verdict, row in sample:
        ws.append([verdict, row["rojak_hits"], None, row["text"]])
    for cell in ws[1]:
        cell.font = Font(bold=True)
    # the filter's own verdict is hidden so it does not bias the labeller
    ws.column_dimensions["A"].hidden = True
    ws.column_dimensions["B"].hidden = True
    ws.column_dimensions["C"].width = 22
    ws.column_dimensions["D"].width = 110
    ws.freeze_panes = "A2"
    dv = DataValidation(type="list", formula1='"Y,N"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add("C2:C%d" % (len(sample) + 1))
    wb.save(CHECK_XLSX)
    print("wrote %d rows -> data/filter_check.xlsx" % len(sample))
    print("Fill the Y/N column, then: python crawl/score_filter.py")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Add new rojak sentences to the corpus.")
    p.add_argument("--min-hits", type=int, default=MIN_ROJAK_HITS,
                   help="rojak markers needed to keep a sentence (default 2)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--validate", action="store_true",
                   help="write a hand-check sample of kept and rejected sentences")
    p.add_argument("--no-malindo", action="store_true",
                   help="skip the dictionary check (faster, much worse)")
    a = p.parse_args()
    build(a.min_hits, a.dry_run, a.validate, not a.no_malindo)
