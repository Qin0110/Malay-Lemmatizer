#!/usr/bin/env python
"""
Dataset REGISTRY for the lemmatizer fine-tune.

    python sources.py --stats            # what is linked, and what it contains
    python sources.py --preview formal   # rendered targets from one source
    python sources.py --build            # write the training/val splits

WHY A REGISTRY AND NOT ONE MERGED FILE
--------------------------------------
The corpora here disagree with each other. malay_lemmatisation_*.jsonl
normalises discourse particles ("la" -> "lah", "je" -> "sahaja");
databuilder.PARTICLES is an explicit do-NOT-normalise set. Merging the raw
files by hand would bake that contradiction into the training data, and it
would be invisible afterwards -- you would only see it as a model that cannot
decide what "la" becomes.

So each corpus stays where it is, on disk, untouched. This module LINKS them:
it declares where each one lives, how to read it, and what policy applies on
the way in. The splits are rebuilt from the sources every time, so fixing a
source or a policy propagates by re-running --build instead of by re-merging.

ADDING A SOURCE
    Append a Source(...) below. `kind` picks the adapter:
      "annotated" -- rows carrying lemmas[{t,l,src}]   (gold, silver, seed)
      "prose"     -- instruction rows explaining each change in words
"""

import argparse
import json
import os
import random
import re
from collections import Counter

import skill
from databuilder import PARTICLES

SEED = 3407
VAL_RATIO = 0.15

OUT_TRAIN = "data/train_linked.jsonl"
OUT_VAL = "data/val_linked.jsonl"
OUT_VAL_SEED = "data/val_linked_seed.jsonl"


class Source:
    def __init__(self, name, path, kind, enabled=True, note=""):
        self.name, self.path, self.kind = name, path, kind
        self.enabled, self.note = enabled, note

    def exists(self):
        return os.path.exists(self.path)


# The three malay_lemmatisation_*.jsonl files are ONE dataset in three
# serialisations -- chat/text/alpaca hold byte-identical (instruction, input,
# output) triples. Linking more than one would train on every example twice,
# so the other two stay declared-but-disabled rather than deleted, to record
# that they were considered.
SOURCES = [
    Source("gold", "../data/gold.jsonl", "annotated",
           note="150 hand-labelled rojak sentences"),
    Source("silver", "../data/silver.jsonl", "annotated",
           note="3000 rule-labelled rojak sentences"),
    Source("seed", "data/val_seed.jsonl", "annotated",
           note="mined sentiment corpus"),
    Source("formal", "../malay_lemmatisation_chat.jsonl", "prose",
           note="1074 formal-Malay sentences w/ per-word explanations"),
    Source("formal_text", "../malay_lemmatisation_text.jsonl", "prose",
           enabled=False, note="SAME DATA as `formal`, ###-instruction format"),
    Source("formal_alpaca", "../malay_lemmatisation_alpaca.jsonl", "prose",
           enabled=False, note="SAME DATA as `formal`, alpaca format"),
]

TOKEN_RE = re.compile(r"[\w'-]+")
# 'N) "token" -> "lemma": reason', up to the next item or the closing sentence.
ITEM_RE = re.compile(
    '\\d+\\)\\s*"([^"]+)"\\s*\u2192\\s*"([^"]+)"\\s*:\\s*'
    '(.*?)(?=\\s*\\d+\\)\\s*"|\\s*The remaining|\\s*$)', re.S)


# ---------------------------------------------------------------------------
# ADAPTERS
# ---------------------------------------------------------------------------

def read_annotated(src):
    """Rows that already carry per-token provenance; skill derives the tag."""
    with open(src.path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                row = json.loads(line)
                row.setdefault("id", "%s_%04d" % (src.name, i))
                row["source"] = src.name
                yield row


def _triple(rec):
    """Pull (instruction, input, output) out of whichever format this is."""
    if "conversations" in rec:
        u = rec["conversations"][0]["content"]
        a = rec["conversations"][1]["content"]
        instr, _, sent = u.partition("\n\nSentence: ")
        return instr.strip(), sent.strip(), a.strip()
    if "instruction" in rec:
        return rec["instruction"].strip(), rec["input"].strip(), rec["output"].strip()
    m = re.search("### Instruction:\n(.*?)\n\n### Input:\n(.*?)\n\n### Response:\n(.*)",
                  rec["text"], re.S)
    return tuple(x.strip() for x in m.groups())


def read_prose(src, stats=None):
    """Turn 'N) "x" -> "y": reason' explanations into per-token rule tags.

    The explanation names only the tokens that CHANGED, which is exactly what
    is needed: anything it does not name is unchanged by definition, so the two
    sentences never have to be aligned against each other. That also handles
    one-to-many rewrites ("Korang" -> "Kamu semua") for free, because the item
    carries the whole multi-word lemma.
    """
    stats = stats if stats is not None else Counter()
    with open(src.path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            _, sentence, answer = _triple(json.loads(line))

            changed = {}
            for tok, lem, why in ITEM_RE.findall(answer):
                key = tok.lower()
                tag = skill.classify_reason(why)
                if tag is None:
                    stats["reason_unclassified"] += 1
                    tag = ("RINGKAS", True)     # a change happened; rule unclear
                if key in changed and changed[key][0].lower() != lem.lower():
                    stats["same_token_two_lemmas"] += 1
                    continue
                changed[key] = (lem, tag[0], tag[1])

            lemmas = []
            for tok in TOKEN_RE.findall(sentence):
                key = tok.lower()
                if key in changed:
                    lem, tag, unsure = changed[key]
                else:
                    # Not named by the explanation, so it was left alone. The
                    # corpus asserts these are "function words or already root
                    # forms", which is what KAMUS means.
                    lem, tag, unsure = tok, "KAMUS", False

                # PARTICLE POLICY. This corpus standardises particles; the rest
                # of the project deliberately does not (databuilder.PARTICLES,
                # and skill.RULES["PARTIKEL"] = "sengaja tidak ditukar"). The
                # project convention wins, so the change is reverted here
                # instead of being allowed to contradict every other source.
                if key in PARTICLES:
                    if lem.lower() != key:
                        stats["particle_reverted"] += 1
                    lem, tag, unsure = tok, "PARTIKEL", False

                lemmas.append({"t": tok, "l": lem, "rule": tag, "unsure": unsure})
                stats[tag] += 1

            if lemmas:
                yield {"id": "%s_%04d" % (src.name, i), "text": sentence,
                       "lemmas": lemmas, "source": src.name}


def load(src, stats=None):
    return read_prose(src, stats) if src.kind == "prose" else read_annotated(src)


def active():
    return [s for s in SOURCES if s.enabled and s.exists()]


# ---------------------------------------------------------------------------

def build():
    os.makedirs("data", exist_ok=True)
    rng = random.Random(SEED)
    train, val, per_source = [], [], Counter()

    for src in active():
        rows = list(load(src))
        rng.shuffle(rows)
        # Split PER SOURCE so every corpus lands on both sides. One global
        # shuffle can leave a small source out of validation entirely, and then
        # its accuracy is simply never measured.
        n_val = max(1, int(len(rows) * VAL_RATIO))
        val += rows[:n_val]
        train += rows[n_val:]
        per_source[src.name] = len(rows)

    rng.shuffle(train)
    rng.shuffle(val)

    def dump(path, rows, fn):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(fn(r), ensure_ascii=False) + "\n")

    dump(OUT_TRAIN, train, skill.to_chatml)
    dump(OUT_VAL, val, skill.to_chatml)
    dump(OUT_VAL_SEED, val, lambda r: r)

    print("linked %d sources" % len(active()))
    for name, n in per_source.most_common():
        print("  %-14s %6d rows" % (name, n))
    print()
    print("  %-14s %6d  ->  %s" % ("train", len(train), OUT_TRAIN))
    print("  %-14s %6d  ->  %s" % ("val", len(val), OUT_VAL))
    print("  %-14s %6d  ->  %s" % ("val (raw)", len(val), OUT_VAL_SEED))
    print("\nTo train on these, set in llm.py:")
    print("  TRAIN_FILE    = %r" % OUT_TRAIN)
    print("  VAL_FILE      = %r" % OUT_VAL)
    print("  VAL_SEED_FILE = %r" % OUT_VAL_SEED)


def stats():
    print("%-15s %-10s %7s  %s" % ("SOURCE", "KIND", "ROWS", "NOTE"))
    print("-" * 92)
    for s in SOURCES:
        if not s.exists():
            print("%-15s %-10s %7s  MISSING: %s" % (s.name, s.kind, "-", s.path))
        elif not s.enabled:
            print("%-15s %-10s %7s  disabled -- %s" % (s.name, s.kind, "-", s.note))
        else:
            print("%-15s %-10s %7d  %s" % (s.name, s.kind, sum(1 for _ in load(s)), s.note))

    tags, notes = Counter(), Counter()
    for s in active():
        for row in load(s, notes):
            for e in row["lemmas"]:
                if e.get("rule"):
                    tags[e["rule"] + ("?" if e.get("unsure") else "")] += 1
                else:
                    t, u = skill.rule_for(e.get("src"), e["t"], e["l"])
                    tags[t + ("?" if u else "")] += 1

    tot = sum(tags.values())
    print("\nrule distribution across all linked sources (%d tokens)" % tot)
    for t, n in tags.most_common():
        print("  %-12s %7d  %5.1f%%" % (t, n, 100.0 * n / tot))

    extra = {k: v for k, v in notes.items() if not k.isupper()}
    if extra:
        print("\nconversion notes")
        for k, v in sorted(extra.items(), key=lambda x: -x[1]):
            print("  %-24s %d" % (k, v))


def preview(name, n):
    src = next((s for s in SOURCES if s.name == name), None)
    if src is None:
        raise SystemExit("unknown source %r; try: %s"
                         % (name, ", ".join(s.name for s in SOURCES)))
    rows = []
    for row in load(src):
        rows.append(row)
        if len(rows) >= n:
            break
    for row in rows:
        print("=" * 72)
        print("INPUT :", row["text"])
        print("-" * 72)
        print(skill.render_target(row))
        print()
    bad = skill._selftest(rows)
    print("self-test: %d/%d render and parse back exactly%s"
          % (len(rows) - len(bad), len(rows), "" if not bad else "  FAILED %s" % bad[:5]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stats", action="store_true")
    p.add_argument("--build", action="store_true")
    p.add_argument("--preview", metavar="SOURCE")
    p.add_argument("-n", type=int, default=3)
    a, _ = p.parse_known_args()

    if a.build:
        build()
    elif a.preview:
        preview(a.preview, a.n)
    else:
        stats()


if __name__ == "__main__":
    main()
