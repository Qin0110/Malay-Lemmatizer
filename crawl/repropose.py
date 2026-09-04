#!/usr/bin/env python
"""
Re-run the pre-fill rules over an already-labelled split, after a rule change.

    python crawl/repropose.py --dry-run              # what would change
    python crawl/repropose.py                        # apply, with a backup
    python crawl/repropose.py --file data/val.jsonl

WHY THIS EXISTS
---------------
When propose() changes, every split labelled before the change is left
describing the old policy. Re-labelling by hand would cost hours and would
mostly reproduce answers the rules can now generate correctly, so the rules
are re-run instead -- but only over tokens no person ever touched.

THE RULE THIS FILE EXISTS TO ENFORCE
------------------------------------
    src == "human" is never overwritten.

An annotator who corrected a token knew something the rules do not. A rule
change is not a reason to discard that; if the new policy disagrees with a
human label, the human label wins and the disagreement is REPORTED so a person
can settle it. Silently overwriting hand-labelling would mean the gold set
slowly becomes a record of the rules rather than an independent check on them,
which is the one thing it cannot be allowed to become.

WHAT CHANGES ARE SAFE
---------------------
This rewrites `l` (the lemma) and `src` (the rule) for machine-labelled
tokens, and rebuilds `lemma_text` to match. It never adds, removes or reorders
tokens, so the alignment every metric depends on is untouched. That invariant
is asserted before anything is written.
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
HUMAN_SRC = "human"


def main(path, dry_run, limit):
    from make_goldsheet import malindo_lemmas, malindo_roots, propose

    if not path.exists():
        sys.exit("No %s" % path)

    print("loading MALINDO...")
    lex, roots = malindo_lemmas(), malindo_roots()
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]

    changed, human_conflicts, moves = [], [], Counter()
    n_tok = 0
    for row in rows:
        for e in row["lemmas"]:
            n_tok += 1
            new_l, new_src = propose(e["t"], lex, roots)

            if e["src"] == HUMAN_SRC:
                # Reported, never applied. See THE RULE above.
                if new_l != e["l"]:
                    human_conflicts.append((e["t"], e["l"], new_l, new_src))
                continue

            if new_l != e["l"] or new_src != e["src"]:
                if new_l != e["l"]:
                    changed.append((e["t"], e["l"], new_l, e["src"], new_src))
                moves["%s -> %s" % (e["src"], new_src)] += 1
                e["l"], e["src"] = new_l, new_src

        row["lemma_text"] = " ".join(e["l"] for e in row["lemmas"])

    print("\n%s: %d sentences, %d tokens" % (path.name, len(rows), n_tok))
    print("lemmas changed: %d (%.1f%%)" % (len(changed),
                                           100 * len(changed) / max(n_tok, 1)))

    if moves:
        print("\nRULE MOVES")
        for move, n in moves.most_common():
            print("  %-28s %5d" % (move, n))

    if changed:
        print("\nCHANGED LEMMAS%s" % ("" if len(changed) <= limit
                                      else " (first %d of %d)" % (limit, len(changed))))
        print("  %-16s %-14s %-16s %s" % ("token", "was", "now", "rule"))
        print("  " + "-" * 62)
        for t, old, new, os_, ns in changed[:limit]:
            print("  %-16s %-14s %-16s %s -> %s" % (t, old, new, os_, ns))

    if human_conflicts:
        print("\nHUMAN LABELS THE NEW POLICY DISAGREES WITH  (%d, NOT changed)"
              % len(human_conflicts))
        print("  Settle these yourself if you care; the hand label stands.")
        print("  %-16s %-14s %s" % ("token", "human said", "rules now say"))
        print("  " + "-" * 52)
        for t, human, new, _src in human_conflicts[:limit]:
            print("  %-16s %-14s %s" % (t, human, new))

    # ---- invariants ------------------------------------------------------
    tok_ok = all(len(r["lemmas"]) == len(r["tokens"]) for r in rows
                 if "tokens" in r)
    txt_ok = all(" ".join(e["l"] for e in r["lemmas"]) == r["lemma_text"]
                 for r in rows if "lemma_text" in r)
    print("\ntoken alignment preserved      %s" % ("PASS" if tok_ok else "FAIL"))
    print("lemma_text reconstructs        %s" % ("PASS" if txt_ok else "FAIL"))
    if not (tok_ok and txt_ok):
        sys.exit("\nInvariant broken - nothing written.")

    if dry_run:
        print("\n--dry-run: nothing written.")
        return
    if not changed and not moves:
        print("\nNothing to change.")
        return

    backup = path.with_suffix(path.suffix + ".pre_repropose")
    if not backup.exists():
        shutil.copy2(path, backup)
        print("\nbacked up -> %s" % backup.name)

    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    # A path passed on the command line need not sit under ROOT, and
    # relative_to() raises rather than falling back when it does not.
    try:
        shown = path.relative_to(ROOT)
    except ValueError:
        shown = path
    print("wrote %s" % shown)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--file", default=str(DEFAULT_FILE),
                   help="labelled split to re-propose (default data/gold.jsonl)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=40,
                   help="how many rows to print (default 40)")
    a = p.parse_args()
    main(Path(a.file), a.dry_run, a.limit)
