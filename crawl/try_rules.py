#!/usr/bin/env python
"""Try the rule system on your own sentences - the CPU counterpart to chat.py.

    python crawl/try_rules.py "aq nk tdr"
    python crawl/try_rules.py                 # interactive
    echo "sy nk yg tu" | python crawl/try_rules.py --stdin

chat.py exercises the fine-tuned model and needs a GPU. Until now there was no
equivalent for the rule cascade: it could only be run through the full
evaluation, or through the Streamlit page. Both are heavy ways to answer "what
does propose() do with this word".

Prints the tag that fired for every token, which is the part worth looking at.
The tag names the branch of the cascade that produced the lemma, so a wrong
answer points straight at the rule that produced it rather than leaving you to
guess which of nine stages was responsible.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "crawl"))

from make_goldsheet import (WORD_RE, malindo_lemmas,        # noqa: E402
                            malindo_roots, pick, propose)

# What each tag means, so the output explains itself.
TAGS = {
    "particle": "a discourse particle (lah, kan, je) stripped",
    "slang": "found in the slang lexicon or BM_dict",
    "redup": "reduplication resolved to its base",
    "elong": "elongated spelling collapsed (sooooo -> so)",
    "affix": "a prefix or suffix stripped",
    "dict": "MALINDO resolved it cleanly",
    "dict_check": "MALINDO echoed an informal spelling back - UNVERIFIED",
    "ambiguous": "several candidates, the first was taken - UNVERIFIED",
    "codeswitch": "detected as English, correctly left alone",
    "num": "a numeral",
    "oov": "nothing in the cascade resolved it - LEFT UNCHANGED",
}


def show(text, lex, roots):
    tokens = WORD_RE.findall(text.lower())
    if not tokens:
        print("  nothing to lemmatise - letters please, not only digits\n")
        return

    rows = []
    for t in tokens:
        lemma, tag = propose(t, lex, roots)
        md, _ = pick(lex.get(t, []), t)
        rows.append((t, lemma, tag, md or t))

    w = max(len(r[0]) for r in rows) + 2
    l = max(len(str(r[1])) for r in rows) + 2
    print("  %-*s %-*s %-12s %s" % (w, "token", l, "rules", "tag", "MALINDO only"))
    print("  " + "-" * (w + l + 40))
    for tok, lemma, tag, md in rows:
        print("  %-*s %-*s %-12s %s" % (w, tok, l, lemma, tag, md))
    print()
    print("  lemma sentence : %s" % " ".join(str(r[1]) for r in rows))
    print("  MALINDO alone  : %s" % " ".join(str(r[3]) for r in rows))

    changed = sum(1 for r in rows if str(r[1]) != r[0])
    missed = sum(1 for r in rows if str(r[1]) != r[3])
    print("  %d of %d tokens changed; MALINDO alone left %d of those unchanged"
          % (changed, len(rows), missed))

    seen = [r[2] for r in rows]
    if seen:
        print()
        print("  tags that fired:")
        for tag in dict.fromkeys(seen):
            flag = " <-- unverified" if tag in ("oov", "dict_check",
                                                "ambiguous") else ""
            print("    %-12s %s%s" % (tag, TAGS.get(tag, "?"), flag))
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("text", nargs="*", help="sentence to lemmatise")
    p.add_argument("--stdin", action="store_true",
                   help="read sentences from stdin, one per line")
    a = p.parse_args()

    print("loading MALINDO... (255,941 rows, about 40s)")
    lex, roots = malindo_lemmas(), malindo_roots()
    print("ready.\n")

    if a.text:
        show(" ".join(a.text), lex, roots)
        return
    if a.stdin:
        for line in sys.stdin:
            if line.strip():
                print("> %s" % line.strip())
                show(line.strip(), lex, roots)
        return

    print("Type a rojak sentence. Blank line or Ctrl+C to quit.\n")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            return
        show(line, lex, roots)


if __name__ == "__main__":
    main()
