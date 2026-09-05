#!/usr/bin/env python
"""
Clean rojak text before it reaches the lemmatiser.

    from preprocess import preprocess, trace
    preprocess("Budak2 tu gilaaa!! 8pagi je")   -> "budak2 tu gila pagi je"

Four stages, in this order: normalise characters, eliminate junk, tokenise,
expand slang. Stage 1 precedes 2 because elongation is measured over adjacent
letters; stage 2 precedes 3 because tokenisation is where the hyphen in
"budak-budak" becomes meaningful.

crawl/evaluate.py deliberately does not call this module: stage 4 resolves
tokens from BM_dict.csv, so scoring through it would measure the dictionary
rather than the weights.
"""

import argparse
import csv
import re
import sys
import unicodedata
from pathlib import Path

LETTER_RUN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_TRAILING_DOUBLE_RE = re.compile(r"(.)\1+$")
_LONG_RUN_RE = re.compile(r"(.)\1{2,}")


def collapse_elongation(word):
    """gilaaa -> gila, gilaa -> gila, sedappp -> sedap, maaf -> maaf."""
    word = _TRAILING_DOUBLE_RE.sub(r"\1", word)
    return _LONG_RUN_RE.sub(r"\1", word)


def normalise_chars(text):
    """Stage 1. Lowercase, then reduce excessive character repetition."""
    text = unicodedata.normalize("NFKC", text)
    return LETTER_RUN_RE.sub(lambda m: collapse_elongation(m.group(0)),
                             text.lower())

NON_LATIN_FLOOR = 0x2E80

URL_RE = re.compile(r"https?://\S+|www\.\S+")
HANDLE_RE = re.compile(r"[@#][\w_]+")


REDUP_MARKER_RE = re.compile(r"^([a-z]{2,})2([^a-z0-9]*)$")

DIGIT_RE = re.compile(r"\d")
DIGIT_TOKEN_RE = re.compile(r"\S*\d\S*")

SYMBOL_RE = re.compile(r"[^a-z\s'-]")
SYMBOL_KEEP_DIGITS_RE = re.compile(r"[^a-z0-9\s'-]")
WS_RE = re.compile(r"\s+")

NUMBER_POLICIES = ("strip", "drop", "keep")


def eliminate(text, number_policy="strip"):
    """Stage 2. Drop emoji, unsupported scripts, URLs, handles and symbols.

        "strip"  drop digit characters, keep the letters  ("8pagi" -> "pagi")
        "drop"   drop any token containing a digit        ("8pagi" -> "")
        "keep"   leave digits alone

    The reduplication marker survives every policy: the "2" in "budak2" is
    shorthand for "budak-budak", not a number.
    """
    if number_policy not in NUMBER_POLICIES:
        raise ValueError("number_policy must be one of %r" % (NUMBER_POLICIES,))

    text = unicodedata.normalize("NFKC", text)
    text = "".join(c for c in text if ord(c) < NON_LATIN_FLOOR)
    text = URL_RE.sub(" ", text)
    text = HANDLE_RE.sub(" ", text)

    out = []
    for tok in text.split():
        m = REDUP_MARKER_RE.match(tok)
        if m:
            out.append(m.group(1) + "2")       # marker kept, punctuation dropped
            continue
        if number_policy == "drop":
            tok = DIGIT_TOKEN_RE.sub(" ", tok)
        elif number_policy == "strip":
            tok = DIGIT_RE.sub("", tok)
        keep = SYMBOL_KEEP_DIGITS_RE if number_policy == "keep" else SYMBOL_RE
        out.append(keep.sub(" ", tok))

    return WS_RE.sub(" ", " ".join(out)).strip()


TOKEN_RE = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)*")


def tokenise(text):
    """Stage 3. budak-budak tu gila -> ['budak-budak', 'tu', 'gila']"""
    return TOKEN_RE.findall(text)

_HERE = Path(__file__).resolve().parent
SLANG_FILE = _HERE / "malindo_morph" / "BM_dict.csv"
USER_FILE = _HERE / "data" / "user_lexicon.csv"
_SLANG = None


def _read_csv(path):
    """{rojak: standard} from a two-column CSV. Missing file is not fatal."""
    out = {}
    try:
        with path.open(encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                key = (row.get("rojak") or "").strip().lower()
                val = (row.get("actual") or "").strip()
                # first spelling wins, matching bm_dict() in make_goldsheet.py
                if key and val and key not in out:
                    out[key] = val
    except OSError:
        pass
    return out


def slang_map():
    global _SLANG
    if _SLANG is None:
        merged = _read_csv(SLANG_FILE)
        merged.update(_read_csv(USER_FILE))       # yours wins
        _SLANG = merged
    return _SLANG


def expand_slang(tokens):
    lookup = slang_map()
    out = []
    for tok in tokens:
        want = lookup.get(tok.lower())
        # split(), so a multi-word expansion stays one token per element
        out.extend(want.split() if want else [tok])
    return out


STAGE_NAMES = ("input", "1 normalisation", "2 elimination", "3 tokens",
               "4 slang")


def trace(text, number_policy="strip", expand=True):
    normalised = normalise_chars(text)
    cleaned = eliminate(normalised, number_policy)
    tokens = tokenise(cleaned)
    expanded = expand_slang(tokens) if expand else tokens
    return list(zip(STAGE_NAMES,
                    [text, normalised, cleaned, tokens, expanded]))


def preprocess(text, number_policy="strip", as_tokens=False, expand=True):
    tokens = tokenise(eliminate(normalise_chars(text), number_policy))
    if expand:
        tokens = expand_slang(tokens)
    return tokens if as_tokens else " ".join(tokens)


DEMO = [
    "Budak2 tu gilaaa!! 3 org je \U0001f602\U0001f602",
    "Zmn skrg org berahsia buat patent cik, dah tak main simpan mulut",
    "aku dh xleh handle keje tu la weh",
    "Jom makan2 kat kedai tu pukul 8pagi, bayar rm3.50 je",
    "Sedappp gilerrr @chef_ali https://t.co/abc #foodie",
    "budak-budak tu gila",
    "Maaf ye, saat tu aku taat sangat สวัสดี",
]


def show(text, number_policy):
    width = max(len(n) for n in STAGE_NAMES)
    for name, value in trace(text, number_policy):
        if isinstance(value, list):
            value = "[%s]" % ", ".join(repr(t) for t in value)
        print("  %-*s  %s" % (width, name, value))
    print()


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser()
    p.add_argument("sentence", nargs="*", help="trace this sentence")
    p.add_argument("--demo", action="store_true", help="run built-in examples")
    p.add_argument("--stdin", action="store_true", help="one sentence per line")
    p.add_argument("--numbers", default="strip", choices=NUMBER_POLICIES,
                   help="what to do with digits (default: strip)")
    a = p.parse_args()

    if a.demo:
        for s in DEMO:
            show(s, a.numbers)
    elif a.stdin:
        for line in sys.stdin:
            if line.strip():
                show(line.strip(), a.numbers)
    elif a.sentence:
        show(" ".join(a.sentence), a.numbers)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
