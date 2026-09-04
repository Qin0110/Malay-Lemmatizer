#!/usr/bin/env python
"""
Bring an old labelled split's English tags up to date. No re-labelling.

    python crawl/retag_english.py --dry-run     # show what would change
    python crawl/retag_english.py               # apply it, with a backup
    python crawl/retag_english.py --file data/val.jsonl

WHAT THIS FIXES
---------------
make_goldsheet.propose() gained an English check (`looks_english`, a Hunspell
en_US lookup) after data/gold.jsonl had already been labelled. Code-switched
tokens in gold therefore fell through to `oov`, while silver.jsonl and the
validation pool -- both built later -- carry `codeswitch` properly:

    silver.jsonl   8,041 codeswitch tokens   (8.3%)   training
    val_pool       144 codeswitch tokens     (7.7%)   validation
    gold.jsonl     0                                  test   <- the odd one out

skill.SRC_TO_RULE maps `codeswitch` to INGGERIS and `oov` to KEKAL. So the
model is trained to answer INGGERIS on English tokens, and then scored against
a test set that calls the same answer KEKAL. That is a scoring artefact, not a
model error, and it would quietly cost the fine-tune several points on the one
tag that makes this a ROJAK lemmatiser rather than a Malay one.

WHAT THIS DOES NOT TOUCH
------------------------
The lemma. Every token retagged here is an English word that keeps its surface
form under either tag, so `l` is unchanged, `lemma_text` is unchanged, and
every lemma-based metric -- accuracy, precision, recall, F1, WER, CER, exact
match -- is bit-for-bit identical before and after. Only the rule-tag metric
moves. That invariant is asserted below, not assumed.

WHY ONLY `oov`
--------------
`codeswitch` is the LAST branch in propose(), reached only after MALINDO has
declined the token. So the tokens that would have become `codeswitch` are
exactly those that ended up `oov`. Tokens tagged `dict`, `dict_check` or
`ambiguous` hit MALINDO first and are none of this function's business --
retagging those would overwrite a real dictionary result with a guess.

`human` is also left alone. An annotator who rewrote a token had a reason the
English check cannot see.
"""

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "crawl"))
sys.path.insert(0, str(ROOT / "unsloth"))

DEFAULT_FILE = ROOT / "data" / "gold.jsonl"

# The only tag eligible for promotion. See WHY ONLY `oov` above.
FROM_SRC = "oov"
TO_SRC = "codeswitch"


def main(path, dry_run):
    from make_goldsheet import looks_english

    if not path.exists():
        sys.exit("No %s" % path)

    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]

    before = Counter()
    changed, skipped, examples = 0, 0, []
    for row in rows:
        for e in row["lemmas"]:
            before[e["src"]] += 1
            if e["src"] != FROM_SRC or not looks_english(e["t"]):
                continue
            # An `oov` token was never resolved, so its lemma should be its own
            # surface form. If it is not, something already rewrote it and this
            # script is not the thing that should have the last word.
            if e["l"].lower() != e["t"].lower():
                skipped += 1
                continue
            e["src"] = TO_SRC
            changed += 1
            if len(examples) < 15:
                examples.append(e["t"])

    after = Counter(e["src"] for r in rows for e in r["lemmas"])
    n_tok = sum(before.values())

    print("%s: %d sentences, %d tokens\n" % (path.name, len(rows), n_tok))
    print("%-12s %8s %8s" % ("src", "before", "after"))
    print("-" * 30)
    for src in sorted(set(before) | set(after)):
        mark = "  <-" if before[src] != after[src] else ""
        print("%-12s %8d %8d%s" % (src, before[src], after[src], mark))

    print("\nretagged %s -> %s : %d tokens (%.1f%%)"
          % (FROM_SRC, TO_SRC, changed, 100 * changed / max(n_tok, 1)))
    if skipped:
        print("skipped %d English `oov` tokens whose lemma differs from the "
              "token - left for a human." % skipped)
    print("examples:", ", ".join(examples))

    # The invariant this whole script rests on.
    lemmas_ok = all(e["l"] for r in rows for e in r["lemmas"])
    rebuilt_ok = all(
        " ".join(e["l"] for e in r["lemmas"]) == r["lemma_text"]
        for r in rows if "lemma_text" in r)
    print("\nlemmas intact                 %s" % ("PASS" if lemmas_ok else "FAIL"))
    print("lemma_text still reconstructs %s"
          % ("PASS" if rebuilt_ok else "FAIL"))
    if not (lemmas_ok and rebuilt_ok):
        sys.exit("\nInvariant broken - nothing written.")

    if dry_run:
        print("\n--dry-run: nothing written.")
        return
    if not changed:
        print("\nNothing to change.")
        return

    backup = path.with_suffix(path.suffix + ".pre_retag")
    if not backup.exists():
        shutil.copy2(path, backup)
        print("\nbacked up -> %s" % backup.name)

    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("wrote %s" % path.relative_to(ROOT))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--file", default=str(DEFAULT_FILE),
                   help="labelled split to retag (default data/gold.jsonl)")
    p.add_argument("--dry-run", action="store_true",
                   help="report the change without writing it")
    a = p.parse_args()
    main(Path(a.file), a.dry_run)
