#!/usr/bin/env python
"""
Clean rojak text before it reaches the lemmatiser.

    from preprocess import preprocess, trace
    preprocess("Budak2 tu gilaaa!! 8pagi je")   -> "budak2 tu gila pagi je"

    python preprocess.py --demo                 # run the built-in examples
    python preprocess.py "Budak2 tu gilaaa!!"   # trace one sentence
    python preprocess.py --stdin < some.txt
    python preprocess.py --numbers drop "bayar rm3.50"

FOUR STAGES, AND WHY THEY ARE IN THIS ORDER
-------------------------------------------
    1. character normalisation   lowercase, then collapse elongation
    2. character elimination     drop emoji, non-Latin scripts, URLs,
                                 @handles, symbols; apply the number policy
    3. tokenisation              split into tokens, keeping hyphenated forms
    4. slang expansion           rewrite known rojak from BM_dict.csv

Stage 1 runs before stage 2 because elongation runs are measured over letters:
"gilaaa!!" only collapses cleanly while the letters are still adjacent to each
other and not yet separated from their punctuation. Lowercasing first means
one rule handles "GILAAA" and "gilaaa" alike.

Stage 2 runs before stage 3 because tokenisation is where the hyphen becomes
meaningful. Eliminating symbols after tokenising would either destroy
"budak-budak" or leave every token carrying trailing junk.

THE REDUPLICATION TRAP
----------------------
The "2" in "budak2" is not a number. It is Malaysian shorthand for
reduplication -- "budak2" means "budak-budak", not "budak two". A naive stage 2
that strips digits turns it into "budak", a different word, with nothing left
on the page to say anything was lost. Every number policy below therefore
protects that marker explicitly before it touches real digits. This is the one
piece of stage 2 that is not a plain regex, and it is the reason why.

WHAT THIS MODULE DOES AND DOES NOT DO
-------------------------------------
Stage 4 expands known slang from the dictionary. Affix stripping and lemma
resolution remain the model's job.

Note what stage 4 means for measurement. Any token the dictionary resolves is
resolved before the model sees it, so accuracy measured over a pipeline that
includes this stage is partly measuring BM_dict.csv rather than the model. That
is why crawl/evaluate.py does not call this module at all -- it feeds the model
the raw sentence, so the reported figures stay a measurement of the weights.
Use expand=False to see what the model does unaided.

Stdlib-only on purpose: this has to stay runnable and testable without torch,
unsloth or a GPU, so the preprocessing can be inspected long before any model
is loaded.
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

    number_policy decides what happens to real digits:

        "strip"  remove digit characters, keep the letters around them
                 "8pagi" -> "pagi",  "2019" -> "",  "rm3.50" -> "rm"
        "drop"   remove any whitespace-token that contains a digit
                 "8pagi" -> "",      "2019" -> "",  "rm3.50" -> ""
        "keep"   leave digits alone

    "strip" is the default because rojak glues counts onto real words
    ("3hari", "2minggu", "8pagi"), and dropping the whole token throws away
    the word the lemmatiser actually needs to see. The cost is that a mostly
    numeric token leaves a stub: "rm3.2k" -> "rmk". Use "drop" when that
    matters more than keeping "pagi".

    The reduplication marker survives every policy, including "drop".
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
    """Stage 3. Split the cleaned sentence into tokens.

    budak-budak tu gila -> ['budak-budak', 'tu', 'gila']
    """
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
                # First spelling of a key wins, matching bm_dict() in
                # make_goldsheet.py so both halves of the project resolve a
                # given token identically.
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
        # split() rather than append(): a multi-word expansion becomes multiple
        # tokens, so every element of the list stays a single token.
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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

    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
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
