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

# ---------------------------------------------------------------------------
# STAGE 4: SLANG EXPANSION
# ---------------------------------------------------------------------------
#
# Rewrites known rojak abbreviations to their standard form from BM_dict.csv:
#
#     dh   -> sudah        kat  -> di
#     nk   -> hendak       xleh -> tidak boleh
#
# Runs LAST, on the token list rather than the raw string. Two reasons: the
# earlier stages have already lowercased and stripped punctuation, so a lookup
# here sees the same key shape the dictionary stores; and an entry that expands
# to several words ("apasal" -> "apa pasal", 10% of the file) has to become
# several tokens rather than one token containing a space, or the token list
# stops being a token list.
#
# That expansion changes the token count, which matters: the model answers with
# one derivation line per token it is given, so it will now describe the
# EXPANDED tokens. Downstream code that assumed the token list matches an
# annotation built from the raw sentence no longer holds.

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
    """The expansion table: BM_dict.csv, overridden by data/user_lexicon.csv.

    The shipped dictionary does not always agree with this project's own
    annotation. It maps `nak` to `hendak` where all 38 occurrences in the gold
    and validation sets say `mahu`, and `x` to `tak` where the labels say
    `tidak`. Left alone, stage 4 would hand the model a form its training data
    calls wrong, on some of the most frequent tokens in the corpus.

    So the user file wins. It is the place to record this project's conventions
    without editing a shipped resource that other parts of the pipeline read.
    """
    global _SLANG
    if _SLANG is None:
        merged = _read_csv(SLANG_FILE)
        merged.update(_read_csv(USER_FILE))       # yours wins
        _SLANG = merged
    return _SLANG


def expand_slang(tokens):
    """budak2 dh gila -> budak2 sudah gila, one token at a time."""
    lookup = slang_map()
    out = []
    for tok in tokens:
        want = lookup.get(tok.lower())
        # split() rather than append(): a multi-word expansion becomes multiple
        # tokens, so every element of the list stays a single token.
        out.extend(want.split() if want else [tok])
    return out


# ---------------------------------------------------------------------------
# THE PIPELINE
# ---------------------------------------------------------------------------

STAGE_NAMES = ("input", "1 normalisation", "2 elimination", "3 tokens",
               "4 slang")


def trace(text, number_policy="strip", expand=True):
    """Run every stage and return [(stage name, value), ...] for display.

    This is what the demo prints and what the report screenshots. Having the
    intermediate values available by name means the pipeline can be shown
    working rather than asserted to work.
    """
    normalised = normalise_chars(text)
    cleaned = eliminate(normalised, number_policy)
    tokens = tokenise(cleaned)
    expanded = expand_slang(tokens) if expand else tokens
    return list(zip(STAGE_NAMES,
                    [text, normalised, cleaned, tokens, expanded]))


def preprocess(text, number_policy="strip", as_tokens=False, expand=True):
    """Full pipeline. Returns a space-joined string unless as_tokens=True.

    The joined string is what gets sent to the model: it was fine-tuned on
    sentences, not on token lists. Returning the list instead means the same
    preprocessing can feed a rule-based lemmatiser, which does want tokens.
    Both come from the same call so the two paths can never drift apart.

    `expand=False` skips the slang stage, which is how to see what the model
    does with raw rojak -- the form it was actually fine-tuned on.
    """
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
