#!/usr/bin/env python
"""
A hand-maintained lexicon applied to the model's answer at inference time.

    python lexicon.py --check tdr bdh nk       # what would each token become?
    python lexicon.py --list                   # everything currently loaded
    python lexicon.py --add tdr tidur          # append to the user lexicon

WHY THIS EXISTS
---------------
The fine-tuned model knows only what it saw in training. A word absent from
the crawled corpus is invisible to it however well documented it is elsewhere:
`tdr` is listed in malindo_morph/BM_dict.csv and appears in zero of the 10,099
corpus sentences, so the rule pipeline resolves it and the model does not.
Retraining to teach one word costs hours and needs corpus sentences containing
it, which is often the thing that does not exist.

This closes that gap without retraining. After the model answers, any token
present in the lexicon has its lemma replaced by the recorded one.

WHAT THIS IS NOT
----------------
It is not a way to make the model look better. Every override is RECORDED and
displayed as coming from the lexicon rather than from the model, because the
derivation format exists so that a reader can tell which component produced
each lemma. Silently overwriting the model's answer would destroy exactly the
property the format was built for.

For the same reason it is OFF in crawl/evaluate.py and cannot be switched on
there. The reported figures measure the model; a lexicon-assisted system is a
different system and would have to be reported as one, with its own row.

PRECEDENCE
----------
    data/user_lexicon.csv        yours, edited by hand -- wins
    malindo_morph/BM_dict.csv    the shipped rojak lexicon, 596 entries

so a wrong entry in the shipped file can be corrected without editing it.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
USER_FILE = ROOT / "data" / "user_lexicon.csv"
SHIPPED_FILE = ROOT / "malindo_morph" / "BM_dict.csv"

HEADER = ("rojak", "actual")
_CACHE = None


def _read(path):
    """{rojak: lemma} from a two-column CSV. Missing file is not an error."""
    out = {}
    if not path.exists():
        return out
    try:
        with path.open(encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                key = (row.get("rojak") or "").strip().lower()
                val = (row.get("actual") or "").strip()
                # First spelling of a key wins, matching bm_dict() in
                # make_goldsheet.py so both halves resolve identically.
                if key and val and key not in out:
                    out[key] = val
    except (OSError, csv.Error) as exc:
        print("[lexicon] could not read %s: %s" % (path.name, exc))
    return out


def load(reload=False):
    """The merged lexicon. User entries override the shipped ones."""
    global _CACHE
    if _CACHE is None or reload:
        merged = _read(SHIPPED_FILE)
        merged.update(_read(USER_FILE))       # user wins
        _CACHE = merged
    return _CACHE


def lookup(token):
    """The recorded lemma for a token, or None."""
    return load().get(token.strip().lower())


def _unresolved(step):
    """Did the model decline to change this token?

    Either it returned the token unchanged, or it tagged the answer KEKAL
    ("unrecognised, left alone"). Both mean the model had nothing to offer,
    which is the only situation where a shipped dictionary should be allowed
    to speak over it.
    """
    return (step["lemma"].strip().lower() == step["token"].strip().lower()
            or step.get("rule") == "KEKAL")


def apply(parsed, rule="LEKSIKON", force_shipped=False):
    """Correct lemmas in a parsed answer from the lexicon, in place.

    TWO SOURCES, TWO LEVELS OF AUTHORITY
    ------------------------------------
    The shipped BM_dict.csv does not always agree with this project's own
    labelling convention. It maps `nk` to `hendak`, while all 462 occurrences
    in the gold, validation and silver sets say `mahu` -- so a blanket override
    would take the dictionary's side against the data the model was trained and
    measured on, and make a correctly-learned token wrong.

    So the two sources are not equal:

        data/user_lexicon.csv   always wins. You wrote it deliberately, and
                                it is the only way to correct a token the model
                                gets confidently wrong.
        malindo_morph/BM_dict   fills gaps only. It speaks where the model left
                                a token unchanged or tagged it KEKAL, and stays
                                quiet where the model committed to an answer.

    `force_shipped` lifts that restriction, for comparing the two behaviours.

    Returns the number of tokens actually changed. Each is marked
    `from_lexicon` and keeps the model's original answer in `model_lemma`, so
    the caller can attribute the correction rather than letting it pass as the
    model's own work.
    """
    user, shipped = _read(USER_FILE), _read(SHIPPED_FILE)
    changed = 0

    for step in parsed.get("steps", []):
        key = step["token"].strip().lower()
        want = user.get(key)
        source = "yours"
        if want is None:
            # Shipped entries only speak into a gap, unless told otherwise.
            if not (force_shipped or _unresolved(step)):
                continue
            want = shipped.get(key)
            source = "shipped"
        if want is None or want.lower() == step["lemma"].strip().lower():
            continue

        step["model_lemma"] = step["lemma"]
        step["lemma"] = want
        step["rule"] = rule
        step["uncertain"] = False
        step["from_lexicon"] = source
        changed += 1

    if changed:
        parsed["sentence"] = " ".join(s["lemma"] for s in parsed["steps"])
    return changed


def add(token, lemma):
    """Append one mapping to the user lexicon, creating the file if needed."""
    token, lemma = token.strip().lower(), lemma.strip()
    if not token or not lemma:
        sys.exit("both a token and a lemma are required")

    existing = _read(USER_FILE)
    if existing.get(token) == lemma:
        print("already present: %s -> %s" % (token, lemma))
        return
    if token in existing:
        print("NOTE: %s is already mapped to %r in %s. Edit that line rather "
              "than appending, or the first one keeps winning."
              % (token, existing[token], USER_FILE.name))
        return

    USER_FILE.parent.mkdir(parents=True, exist_ok=True)
    new = not USER_FILE.exists() or USER_FILE.stat().st_size == 0
    with USER_FILE.open("a", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(HEADER)
        w.writerow([token, lemma])
    load(reload=True)

    shipped = _read(SHIPPED_FILE).get(token)
    print("added: %s -> %s   (%s)" % (token, lemma, USER_FILE.name))
    if shipped and shipped != lemma:
        print("       overrides the shipped entry %s -> %s" % (token, shipped))


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--check", nargs="+", metavar="TOKEN",
                   help="what the lexicon would make of these tokens")
    p.add_argument("--add", nargs=2, metavar=("TOKEN", "LEMMA"),
                   help="append a mapping to the user lexicon")
    p.add_argument("--list", action="store_true",
                   help="print the user lexicon and the totals")
    a = p.parse_args()

    if a.add:
        add(*a.add)
        return
    if a.check:
        user, shipped = _read(USER_FILE), _read(SHIPPED_FILE)
        for t in a.check:
            k = t.strip().lower()
            if k in user:
                print("  %-12s -> %-14s (yours)" % (t, user[k]))
            elif k in shipped:
                print("  %-12s -> %-14s (shipped)" % (t, shipped[k]))
            else:
                print("  %-12s -> %-14s add it with --add"
                      % (t, "(not known)"))
        return
    if a.list:
        user, shipped = _read(USER_FILE), _read(SHIPPED_FILE)
        print("%s  %d entries" % (SHIPPED_FILE.name, len(shipped)))
        print("%s  %d entries" % (USER_FILE.name, len(user)))
        if user:
            print()
            for k in sorted(user):
                over = shipped.get(k)
                mark = ("   overrides %r" % over) if over and over != user[k] else ""
                print("  %-14s -> %s%s" % (k, user[k], mark))
        print("\nmerged total: %d" % len(load(reload=True)))
        return
    p.print_help()


if __name__ == "__main__":
    main()
