#!/usr/bin/env python
"""Try the rules on your own sentences (no GPU).

    python crawl/try_rules.py "aq nk tdr dlu, esok kite jumpe kat umah"
    python crawl/try_rules.py                  interactive
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rules                                                      # noqa: E402
from preprocess import prepare                                    # noqa: E402

TAGS = {
    "particle": "discourse particle, kept",
    "slang": "slang lexicon",
    "name": "proper noun, kept as written",
    "redup": "reduplication resolved to its base",
    "elong": "elongation collapsed",
    "affix": "affix stripped (MALINDO root)",
    "deriv": "derived noun, kept",
    "dict": "MALINDO, already a lemma",
    "dict_check": "MALINDO only echoed the spelling - unverified",
    "ambiguous": "several MALINDO lemmas, shortest taken - unverified",
    "codeswitch": "English, kept",
    "num": "number",
    "oov": "nothing matched, kept - unverified",
}


def show(text):
    lex = rules.malindo()[0]
    for sent in prepare(text):
        labels = rules.label(sent)
        w = max(len(e["t"]) for e in labels) + 2
        print("  %-*s %-16s %-11s %s" % (w, "token", "lemma", "rule", "MALINDO only"))
        print("  " + "-" * (w + 45))
        for e in labels:
            low = e["t"].lower()
            plain, _ = rules.pick(lex.get(low, []), low)
            print("  %-*s %-16s %-11s %s" % (w, e["t"], e["l"], e["src"], plain or low))
        print("\n  " + rules.lemma_text(labels))
        print("  tags: " + "; ".join("%s = %s" % (t, TAGS.get(t, "?"))
                                     for t in dict.fromkeys(e["src"] for e in labels)))
        print()


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="Run the rule lemmatiser.")
    p.add_argument("text", nargs="*")
    a = p.parse_args()

    print("loading MALINDO...")
    rules.malindo()
    if a.text:
        show(" ".join(a.text))
        return
    print("Type a sentence. Blank line to quit.\n")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if not line:
            return
        show(line)


if __name__ == "__main__":
    main()
