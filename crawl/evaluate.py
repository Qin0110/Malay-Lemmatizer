#!/usr/bin/env python
"""Score the lemmatisers on the hand-labelled sets.

    python crawl/evaluate.py                      rules only, gold test set (no GPU)
    python crawl/evaluate.py --val                the validation set instead
    python crawl/evaluate.py --model llama-ft     + the model, and model + rules
    python crawl/evaluate.py --model llama-ft --model qwen-ft --val
    python crawl/evaluate.py --reuse --model llama-ft    re-score saved outputs, no GPU

Pick between models on --val. Use the gold set once, for the model you chose.
"""

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pipeline                                                   # noqa: E402
import rules                                                      # noqa: E402
import skill                                                      # noqa: E402

DATA = ROOT / "data"
SPLITS = {"gold": DATA / "gold.jsonl", "val": DATA / "val.jsonl"}
PRED_DIR = DATA / "predictions"


def norm(w):
    return re.sub(r"[^\w]", "", w.lower())


def jl(path):
    with path.open(encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def load_rows(split):
    if split == "dev":
        man = json.loads((DATA / "split_manifest.json").read_text(encoding="utf-8"))
        dev = set(man["dev_ids"])
        return [r for r in jl(DATA / "silver.jsonl") if r["id"] in dev]
    if not SPLITS[split].exists():
        sys.exit("No %s." % SPLITS[split].relative_to(ROOT))
    return jl(SPLITS[split])


def align_lcs(pred_words, lemmas):
    """Map predicted words onto the gold tokens; None where nothing matched."""
    gold_words, owner = [], []
    for i, e in enumerate(lemmas):
        for w in e["l"].split():
            gold_words.append(norm(w))
            owner.append(i)
    sm = difflib.SequenceMatcher(None, gold_words, [norm(w) for w in pred_words],
                                 autojunk=False)
    matched = [False] * len(gold_words)
    got = [None] * len(lemmas)
    for block in sm.get_matching_blocks():
        for k in range(block.size):
            gi, pi = block.a + k, block.b + k
            matched[gi] = True
            tok = owner[gi]
            got[tok] = (got[tok] + " " if got[tok] else "") + pred_words[pi]
    for i in range(len(lemmas)):
        if not all(matched[j] for j, o in enumerate(owner) if o == i):
            got[i] = None
    return got


def fit(lemmas, row):
    """One prediction per gold token."""
    if len(lemmas) == len(row["lemmas"]):
        return lemmas
    return align_lcs(lemmas, row["lemmas"])


def predict_rules(row):
    return fit([e["l"] for e in rules.label(row["text"])], row)


def predict_model(row, raw):
    parsed = skill.parse_response(raw)
    if parsed["steps"]:
        return fit([s["lemma"] for s in parsed["steps"]], row), parsed["ok"]
    return align_lcs(raw.split(), row["lemmas"]), False


def predict_merged(row, raw):
    parsed = skill.parse_response(raw)
    steps = pipeline.merge(pipeline.rule_steps(row["text"]), parsed, True)
    return fit([s["lemma"] for s in steps], row)


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def score(rows, preds):
    s = {"hit": 0, "n": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0,
         "human_hit": 0, "human_n": 0, "w_err": 0, "w_len": 0,
         "c_err": 0, "c_len": 0}
    for row, pred in zip(rows, preds):
        for e, p in zip(row["lemmas"], pred):
            hit = p is not None and norm(p) == norm(e["l"])
            changed = norm(e["t"]) != norm(e["l"])
            acted = p is not None and norm(p) != norm(e["t"])
            if changed and acted and hit:
                s["tp"] += 1
            elif changed and not acted:
                s["fn"] += 1
            elif changed and acted:
                s["fp"] += 1
                s["fn"] += 1
            elif acted:
                s["fp"] += 1
            else:
                s["tn"] += 1
            s["n"] += 1
            s["hit"] += hit
            if e["src"] == "human":
                s["human_n"] += 1
                s["human_hit"] += hit

        ref = [norm(w) for e in row["lemmas"] for w in e["l"].split() if norm(w)]
        hyp = [norm(w) for p in pred if p for w in p.split() if norm(w)]
        s["w_err"] += edit_distance(ref, hyp)
        s["w_len"] += len(ref)
        s["c_err"] += edit_distance("".join(ref), "".join(hyp))
        s["c_len"] += len("".join(ref))
    return s


def table(results):
    print("=" * 92)
    print("%-16s %9s %10s %8s %8s %10s %8s %8s %9s"
          % ("system", "accuracy", "precision", "recall", "F1",
             "corrected", "WER", "CER", "fmt-fail"))
    print("-" * 92)
    for name, s, fail in results:
        p = s["tp"] / (s["tp"] + s["fp"]) if s["tp"] + s["fp"] else 0
        r = s["tp"] / (s["tp"] + s["fn"]) if s["tp"] + s["fn"] else 0
        f = 2 * p * r / (p + r) if p + r else 0
        human = ("%9.1f%%" % (100 * s["human_hit"] / s["human_n"])
                 if s["human_n"] else "%10s" % "-")
        print("%-16s %8.1f%% %9.1f%% %7.1f%% %7.1f%% %s %8.4f %8.4f %9s"
              % (name, 100 * s["hit"] / s["n"], 100 * p, 100 * r, 100 * f, human,
                 s["w_err"] / max(s["w_len"], 1), s["c_err"] / max(s["c_len"], 1),
                 "-" if fail is None else "%.1f%%" % fail))
    print("=" * 92)
    print("corrected = accuracy on the tokens a person changed when labelling; "
          "the other labels started as rule output.")


def model_outputs(name, rows, split, reuse):
    cache = PRED_DIR / ("%s_%s.json" % (split, name))
    if reuse:
        if not cache.exists():
            print("no saved outputs for %s on %s - run without --reuse first\n"
                  % (name, split))
            return None
        saved = json.loads(cache.read_text(encoding="utf-8"))
        missing = [r["id"] for r in rows if r["id"] not in saved]
        if missing:
            print("saved outputs for %s are missing %d sentences - rerun it\n"
                  % (name, len(missing)))
            return None
        return [saved[r["id"]] for r in rows]

    if not pipeline.gpu_packages():
        sys.exit("The GPU packages are not installed (SETUP.bat gpu).")
    print("running %s on %d sentences..." % (name, len(rows)))
    model = pipeline.Model(name)
    out = []
    for i, row in enumerate(rows, 1):
        out.append(model.generate(row["text"]))
        if i % 25 == 0:
            print("   %d/%d" % (i, len(rows)))
    model.unload()
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({r["id"]: t for r, t in zip(rows, out)},
                                ensure_ascii=False, indent=1), encoding="utf-8")
    print("   saved -> %s\n" % cache.relative_to(ROOT))
    return out


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(description="Score the lemmatisers.")
    p.add_argument("--val", action="store_true", help="use the validation set")
    p.add_argument("--dev", action="store_true", help="use the (rule-labelled) dev split")
    p.add_argument("--model", action="append", default=[],
                   help="llama-ft, qwen-ft, llama-base, qwen-base or an adapter folder")
    p.add_argument("--reuse", action="store_true",
                   help="score saved model outputs instead of running the model")
    a = p.parse_args()
    if a.val and a.dev:
        p.error("pick one of --val and --dev")

    split = "val" if a.val else "dev" if a.dev else "gold"
    rows = load_rows(split)
    n_tok = sum(len(r["lemmas"]) for r in rows)
    n_chg = sum(1 for r in rows for e in r["lemmas"] if norm(e["t"]) != norm(e["l"]))
    print("%s: %d sentences, %d tokens, %d need changing (%.0f%%)\n"
          % (split, len(rows), n_tok, n_chg, 100 * n_chg / n_tok))
    if split == "dev":
        print("Note: dev labels come from the rules, so the rules score ~100% here.\n")

    print("loading MALINDO...")
    results = [("rules", score(rows, [predict_rules(r) for r in rows]), None)]

    for name in a.model:
        outputs = model_outputs(name, rows, split, a.reuse)
        if outputs is None:
            continue
        preds, fails = zip(*(predict_model(r, o) for r, o in zip(rows, outputs)))
        fail = 100.0 * sum(1 for ok in fails if not ok) / len(rows)
        results.append((name, score(rows, preds), fail))
        merged = [predict_merged(r, o) for r, o in zip(rows, outputs)]
        results.append((name + " + rules", score(rows, merged), None))

    print()
    table(results)


if __name__ == "__main__":
    main()
