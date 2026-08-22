#!/usr/bin/env python
"""
Clean rojak text before it reaches the lemmatiser.

    from preprocess import preprocess, trace
    preprocess("Budak2 tu gilaaa!! 3 org je")   -> "budak-budak tu gila org je"

    python preprocess.py --demo                  # run the built-in examples
    python preprocess.py "Budak2 tu gilaaa!!"    # trace one sentence
    python preprocess.py --stdin < some.txt

FOUR STAGES, AND WHY THEY ARE IN THIS ORDER
-------------------------------------------
    1. character normalisation   lowercase, then collapse elongation
    2. reduplication expansion   budak2 -> budak-budak
    3. regex elimination         drop emoji, symbols, digits
    4. tokenisation              split into a token list

Stage 2 MUST run before stage 3. The "2" in "budak2" is not a number, it is
the reduplication marker. Eliminate digits first and "budak2" quietly becomes
"budak" -- a different word, with no trace left that anything was lost.

Stage 1 must run before stage 2 for the same class of reason: "budakk2" is an
elongated reduplication, and only collapsing the "kk" first lets stage 2 see
the "budak2" it knows how to expand.

Elongation is applied to each run of letters rather than to the whole string,
because databuilder.normalise_repeat anchors one of its two rules to the end
of a word ("gilaa" -> "gila"). Run it over a sentence and that anchor only
ever matches the final word; run it over "gilaa!!" and the punctuation hides
the run entirely. Letter runs sidestep both.

WHAT THIS MODULE DELIBERATELY DOES NOT DO
-----------------------------------------
No slang expansion, no affix stripping, no dictionary lookup. Those are the
model's job -- that is what it was fine-tuned for. Doing them here would
double-handle the tokens the fine-tune exists to handle, and would make it
impossible to tell which component produced a given lemma.

REDUPLICATION: EXPAND, NOT RESOLVE
----------------------------------
This module EXPANDS ("budak2" -> "budak-budak"), matching
Preprocess/normalizer.py. databuilder.lemma_of instead RESOLVES to the base
("budak2" -> "budak"). Both are fine here: lemma_of already handles the
hyphenated form as reduplication too, so expanding first and resolving later
lands on the same lemma. See the NOTE on GANDA in skill.py -- the two halves
of this project disagree about which form is the lemma, and that disagreement
is still open.

Stdlib-only on purpose, like skill.py: this has to stay runnable and testable
without unsloth, torch or a GPU.
"""

import argparse
import re
import sys
import unicodedata

# Reused rather than re-derived, so inference collapses elongation exactly the
# way the training data was built. databuilder is stdlib-only and does no work
# at import time, so this costs nothing.
from databuilder import TOKEN_RE, normalise_repeat

# ---------------------------------------------------------------------------
# STAGE 1: CHARACTER NORMALISATION
# ---------------------------------------------------------------------------

# A maximal run of letters, with no digits and no underscore. Applying
# normalise_repeat per run keeps its end-of-word rule meaningful inside a
# sentence and immune to trailing punctuation.
LETTER_RUN_RE = re.compile(r"[^\W\d_]+")


def normalise_chars(text):
    """Lowercase, then collapse elongated letters.

    gilaaa -> gila, sedappp -> sedap, sabunnn, -> sabun,
    """
    return LETTER_RUN_RE.sub(
        lambda m: normalise_repeat(m.group(0)) or m.group(0),
        text.lower(),
    )


# ---------------------------------------------------------------------------
# STAGE 2: REDUPLICATION
# ---------------------------------------------------------------------------

# Same shape as Preprocess/normalizer.REDUP_RE, lowercased because stage 1 has
# already run. The trailing group keeps punctuation attached ("budak2," works).
REDUP_RE = re.compile(r"^([a-z]{2,})2([^a-z0-9]*)$")


def expand_reduplication(text):
    """budak2 -> budak-budak, makan2! -> makan-makan!"""
    out = []
    for word in text.split():
        m = REDUP_RE.match(word)
        if m:
            base, tail = m.group(1), m.group(2)
            word = "%s-%s%s" % (base, base, tail)
        out.append(word)
    return " ".join(out)


# ---------------------------------------------------------------------------
# STAGE 3: REGEX ELIMINATION
# ---------------------------------------------------------------------------

# databuilder.clean uses the same cut-off: everything from CJK upwards goes,
# which covers emoji, kaomoji blocks and pictographs in one comparison.
NON_LATIN_FLOOR = 0x2E80

URL_RE = re.compile(r"https?://\S+|www\.\S+")
HANDLE_RE = re.compile(r"[@#][\w_]+")
DIGIT_RE = re.compile(r"\d")
DIGIT_WORD_RE = re.compile(r"\S*\d\S*")

# Keep letters, whitespace, apostrophe and hyphen. The hyphen has to survive:
# stage 2 just used it to write the reduplication out in full.
SYMBOL_RE = re.compile(r"[^a-z\s'-]")
SYMBOL_KEEP_NUM_RE = re.compile(r"[^a-z0-9\s'-]")
WS_RE = re.compile(r"\s+")

NUMBER_POLICIES = ("strip", "drop", "keep")


def eliminate(text, number_policy="strip"):
    """Drop emoji, symbols and numbers.

    number_policy:
        "strip"  remove digit characters, keep the letters around them
                 ("8pagi" -> "pagi", "2019" -> "")
        "drop"   remove any whitespace-token containing a digit
                 ("8pagi" -> "", "2019" -> "")
        "keep"   leave digits alone

    "strip" is the default because rojak glues counts onto real words
    ("3hari", "2minggu", "8pagi") and dropping the whole token throws away the
    word the lemmatiser actually needs. The cost is that a token which is
    mostly digits can leave a stub behind: "rm3.2k" -> "rmk". Use "drop" if
    that matters more than keeping "pagi".
    """
    if number_policy not in NUMBER_POLICIES:
        raise ValueError("number_policy must be one of %r" % (NUMBER_POLICIES,))

    text = unicodedata.normalize("NFKC", text)
    text = "".join(c for c in text if ord(c) < NON_LATIN_FLOOR)
    text = URL_RE.sub(" ", text)
    text = HANDLE_RE.sub(" ", text)

    if number_policy == "drop":
        text = DIGIT_WORD_RE.sub(" ", text)
    elif number_policy == "strip":
        text = DIGIT_RE.sub("", text)

    keep = SYMBOL_KEEP_NUM_RE if number_policy == "keep" else SYMBOL_RE
    text = keep.sub(" ", text)
    return WS_RE.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# STAGE 4: TOKENISATION
# ---------------------------------------------------------------------------

def tokenise(text):
    """Split into tokens.

    databuilder.TOKEN_RE is reused so a token here means the same thing it
    meant when the corpus was annotated. It treats "budak-budak" as one token,
    which is what stage 2 was building.
    """
    return TOKEN_RE.findall(text)


# ---------------------------------------------------------------------------
# THE PIPELINE
# ---------------------------------------------------------------------------

STAGE_NAMES = ("input", "1 characters", "2 reduplication",
               "3 elimination", "4 tokens")


def trace(text, number_policy="strip"):
    """Run every stage and return [(stage name, value), ...] for display."""
    chars = normalise_chars(text)
    redup = expand_reduplication(chars)
    clean = eliminate(redup, number_policy)
    tokens = tokenise(clean)
    return list(zip(STAGE_NAMES, [text, chars, redup, clean, tokens]))


def preprocess(text, number_policy="strip", as_tokens=False):
    """Full pipeline. Returns a space-joined string unless as_tokens=True.

    The joined string is what chat.py sends to the model: the fine-tune reads
    a sentence, not a token list, so the tokens are put back together with
    single spaces. Returning the list as well means the same preprocessing can
    feed a rule-based lemmatiser that does want tokens.
    """
    tokens = tokenise(eliminate(
        expand_reduplication(normalise_chars(text)), number_policy))
    return tokens if as_tokens else " ".join(tokens)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEMO = [
    "Budak2 tu gilaaa!! 3 org je 😂😂",
    "Zmn skrg org berahsia buat patent cik, dah tak main simpan mulut",
    "aku dh xleh handle keje tu la weh",
    "Jom makan2 kat kedai tu pukul 8pagi, bayar rm3.50 je",
    "Sedappp gilerrr @chef_ali https://t.co/abc #foodie",
]


def show(text, number_policy):
    width = max(len(n) for n in STAGE_NAMES)
    for name, value in trace(text, number_policy):
        if isinstance(value, list):
            value = "[%s]" % ", ".join(repr(t) for t in value)
        print("  %-*s  %s" % (width, name, value))
    print()


def main():
    # The demo inputs contain emoji by design, and the default Windows console
    # codepage (cp1252) raises on them mid-print. Only the CLI needs this --
    # importers get the functions without the side effect.
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
