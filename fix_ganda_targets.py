#!/usr/bin/env python
"""
Teach the SFT targets the kata ganda semu exception.

    python fix_ganda_targets.py                 # writes *.fixed.jsonl, reports
    python fix_ganda_targets.py --in-place      # overwrite the input
    python fix_ganda_targets.py --show 20       # print the first 20 edits

The targets in train_sft.jsonl resolve every reduplication to its base, so
"tiba2" is labelled "tiba" and "masing2" is labelled "masing". For an ordinary
plural that is right -- "budak2" really is "budak" -- but kata ganda semu is
only ever a word in its doubled form, and collapsing it destroys the word.
A model trained on those targets learns "always collapse", which is why it
cannot keep "rama-rama" together.

This rewrites only the affected targets, using malindo_morph/ganda_semu.txt as
the list of forms that must survive doubled. Everything else is left alone.

Ambiguity is reported, not guessed: if the user turn has one "tiba2" but the
target contains "tiba" twice, there is no way to tell which occurrence came
from the reduplication, so the row is skipped and counted under `ambiguous`.
"""

import argparse
import json
import re

from malindo_morph.dictionary import Dictionary

DIGIT2_RE = re.compile(r"\b([A-Za-z]{2,})2\b")
DOUBLED_RE = re.compile(r"\b([A-Za-z]{2,})-([A-Za-z]{2,})\b")


def wanted_forms(user_turn, index, keep):
    """(base, doubled form) for each ganda semu the user turn contains."""
    out = {}
    for base in DIGIT2_RE.findall(user_turn):
        base = base.lower()
        cands = index.get((base, ""))
        full = (max(cands, key=lambda f: (cands[f], f == base + "-" + base))
                if cands else base + "-" + base)
        if full in keep:
            out.setdefault(full, base)
    for a, b in DOUBLED_RE.findall(user_turn):
        form = (a + "-" + b).lower()
        if form in keep and a.lower() == b.lower():
            out.setdefault(form, a.lower())
    return out


def fix_target(user_turn, target, index, keep):
    """Return (new target, edits, ambiguous). Target is unchanged if no edits."""
    edits, ambiguous = [], []
    for full, base in wanted_forms(user_turn, index, keep).items():
        if re.search(r"\b%s\b" % re.escape(full), target):
            continue                                  # already correct
        occurrences = re.findall(r"\b%s\b" % re.escape(base), target)
        if not occurrences:
            continue                                  # target dropped the word
        # One bare base per reduplication in the source, or there is no way to
        # know which occurrence the reduplication produced.
        if len(occurrences) != len(re.findall(r"\b%s2\b" % re.escape(base), user_turn)):
            ambiguous.append((base, full))
            continue
        target = re.sub(r"\b%s\b" % re.escape(base), full, target)
        edits.append((base, full))
    return target, edits, ambiguous


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="unsloth/data/train_sft.jsonl")
    p.add_argument("--output", default=None,
                   help="default: <input> with .fixed.jsonl")
    p.add_argument("--in-place", action="store_true")
    p.add_argument("--show", type=int, default=8, help="how many edits to print")
    a = p.parse_args()

    out_path = (a.input if a.in_place
                else a.output or a.input.replace(".jsonl", ".fixed.jsonl"))

    dictionary = Dictionary()
    index, keep = dictionary.reduplication_index(), dictionary.ganda_semu
    print("keep-list: %d forms" % len(keep))

    with open(a.input, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]

    changed = shown = 0
    edit_counts, ambiguous_counts = {}, {}
    for row in rows:
        user = row["messages"][1]["content"]
        target = row["messages"][2]["content"]
        new, edits, ambiguous = fix_target(user, target, index, keep)
        for base, full in ambiguous:
            ambiguous_counts[full] = ambiguous_counts.get(full, 0) + 1
        if not edits:
            continue
        changed += 1
        for base, full in edits:
            edit_counts[full] = edit_counts.get(full, 0) + 1
        if shown < a.show:
            print("\nUSER  : %s" % user[:96])
            print("BEFORE: %s" % target[:96])
            print("AFTER : %s" % new[:96])
            shown += 1
        row["messages"][2]["content"] = new

    with open(out_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\n%d of %d rows changed -> %s" % (changed, len(rows), out_path))
    print("\nedits by form:")
    for form, n in sorted(edit_counts.items(), key=lambda kv: -kv[1]):
        print("  %-18s %d" % (form, n))
    if ambiguous_counts:
        print("\nskipped as ambiguous (base appears more often than the "
              "reduplication does, so the right occurrence is unknowable):")
        for form, n in sorted(ambiguous_counts.items(), key=lambda kv: -kv[1]):
            print("  %-18s %d" % (form, n))


if __name__ == "__main__":
    main()
