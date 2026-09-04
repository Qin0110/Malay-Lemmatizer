#!/usr/bin/env python
"""
Clean rojak text before it reaches the lemmatiser.

    from preprocess import preprocess, trace
    preprocess("Budak2 tu gilaaa!! 8pagi je")   -> "budak2 tu gila pagi je"

    python preprocess.py --demo                 # run the built-in examples
    python preprocess.py "Budak2 tu gilaaa!!"   # trace one sentence
    python preprocess.py --stdin < some.txt
    python preprocess.py --numbers drop "bayar rm3.50"

THREE STAGES, AND WHY THEY ARE IN THIS ORDER
--------------------------------------------
    1. character normalisation   lowercase, then collapse elongation
    2. character elimination     drop emoji, non-Latin scripts, URLs,
                                 @handles, symbols; apply the number policy
    3. tokenisation              split into tokens, keeping hyphenated forms

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

WHAT THIS MODULE DELIBERATELY DOES NOT DO
-----------------------------------------
No slang expansion, no affix stripping, no dictionary lookup. Those are the
lemmatiser's job. Doing them here would mean the reported model accuracy is
partly measuring this file, and there would be no way to tell which component
produced a given lemma.

Stdlib-only on purpose: this has to stay runnable and testable without torch,
unsloth or a GPU, so the preprocessing can be inspected long before any model
is loaded.
"""

import argparse
import re
import sys
import unicodedata

# ---------------------------------------------------------------------------
# STAGE 1: CHARACTER NORMALISATION
# ---------------------------------------------------------------------------
#
# Lowercase, then reduce excessive character repetition:
#     gilaaa   -> gila
#     sedappp  -> sedap
#     GILAAA   -> gila
#
# Two rules, applied in this order, because they answer different questions:
#
#   (a) a run of 3+ identical letters anywhere is always emphasis. No Malay
#       root triples a letter, so collapsing to one is safe.
#   (b) a run of exactly 2 identical letters AT THE END of a word is usually
#       emphasis too ("gilaa", "okk"), but the same doubling inside a word is
#       often real ("maaf", "taat", "saat"). Anchoring rule (b) to the end of
#       the word is what keeps "maaf" intact.
#
# Applied per letter-run rather than per whole string: run it over the whole
# sentence and rule (b)'s end anchor would only ever match the final word.

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


# ---------------------------------------------------------------------------
# STAGE 2: CHARACTER ELIMINATION
# ---------------------------------------------------------------------------
#
# Remove emoji, unsupported scripts, URLs, user handles and unwanted symbols,
# then apply the selected number-handling policy.

# Everything from U+2E80 (CJK radicals) upwards goes in one comparison: CJK,
# Thai sits below it so is handled by the symbol pass, and the whole emoji /
# pictograph / kaomoji range sits above. Cheaper and more readable than a
# hand-maintained list of emoji blocks.
NON_LATIN_FLOOR = 0x2E80

URL_RE = re.compile(r"https?://\S+|www\.\S+")
HANDLE_RE = re.compile(r"[@#][\w_]+")

# The reduplication marker, with any trailing punctuation. See THE
# REDUPLICATION TRAP in the module docstring.
REDUP_MARKER_RE = re.compile(r"^([a-z]{2,})2([^a-z0-9]*)$")

DIGIT_RE = re.compile(r"\d")
DIGIT_TOKEN_RE = re.compile(r"\S*\d\S*")

# What survives: letters, whitespace, apostrophe, hyphen. The apostrophe is
# kept for "do'a" / "ma'af"; the hyphen is kept because stage 3 needs it to
# recognise "budak-budak" as one token.
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


# ---------------------------------------------------------------------------
# STAGE 3: TOKENISATION
# ---------------------------------------------------------------------------
#
# Split on whitespace boundaries while keeping the internal punctuation that
# carries meaning. A hyphen is PERMITTED between two alphanumeric runs, so
# "budak-budak" stays one token, but a dangling "-" or a run of "---" does not
# glue anything together and is dropped.
#
# One token here must mean one line in the model's answer, because the
# evaluation aligns the model's output against this token list positionally.
# A tokenisation change is therefore a scoring change; do not make one without
# regenerating the annotations.

TOKEN_RE = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)*")


def tokenise(text):
    """Stage 3. Split the cleaned sentence into tokens.

    budak-budak tu gila -> ['budak-budak', 'tu', 'gila']
    """
    return TOKEN_RE.findall(text)


# ---------------------------------------------------------------------------
# THE PIPELINE
# ---------------------------------------------------------------------------

STAGE_NAMES = ("input", "1 normalisation", "2 elimination", "3 tokens")


def trace(text, number_policy="strip"):
    """Run every stage and return [(stage name, value), ...] for display.

    This is what the demo prints and what the report screenshots. Having the
    intermediate values available by name means the pipeline can be shown
    working rather than asserted to work.
    """
    normalised = normalise_chars(text)
    cleaned = eliminate(normalised, number_policy)
    tokens = tokenise(cleaned)
    return list(zip(STAGE_NAMES, [text, normalised, cleaned, tokens]))


def preprocess(text, number_policy="strip", as_tokens=False):
    """Full pipeline. Returns a space-joined string unless as_tokens=True.

    The joined string is what gets sent to the model: it was fine-tuned on
    sentences, not on token lists. Returning the list instead means the same
    preprocessing can feed a rule-based lemmatiser, which does want tokens.
    Both come from the same call so the two paths can never drift apart.
    """
    tokens = tokenise(eliminate(normalise_chars(text), number_policy))
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
    # The demo inputs contain emoji by design, and the default Windows console
    # codepage raises on them mid-print. Only the CLI needs this -- importers
    # get the functions without the side effect.
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
