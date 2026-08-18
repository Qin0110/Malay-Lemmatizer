#!/usr/bin/env python
"""
Score the lemmatisers against the hand-labelled gold set.

    python crawl/evaluate.py                 # baselines only, no GPU needed
    python crawl/evaluate.py --model         # + the fine-tuned adapter

WHAT IS BEING COMPARED

    copy         return every token unchanged. Not a real system - it is the
                 floor. 67% of gold tokens are already their own lemma, so
                 any result below this is worse than doing nothing.

    malindo      MALINDO Morph alone, first lemma, no slang lexicon. This is
                 the pipeline in main.py / streamlit.py - a plain dictionary
                 lemmatiser, the honest "what you get for free" baseline.

    rules        MALINDO root column + the SLANG lexicon + orthographic
                 repair (elongation, reduplication). The strongest baseline
                 buildable without machine learning, and the one the
                 fine-tune has to beat to be worth anything.

    finetune     the QLoRA adapter in unsloth/lemmatizer_lora.

WHY TWO NUMBERS PER SYSTEM

    Overall accuracy is dominated by tokens that need no change at all. The
    honest measure is accuracy on the CHANGED subset - tokens whose gold
    lemma differs from the surface form. That is the work. A system can post
    85% overall and still be useless if all of it comes from the free 67%.
"""

import argparse
import difflib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "crawl"))
sys.path.insert(0, str(ROOT / "unsloth"))

from make_goldsheet import (malindo_lemmas, malindo_roots,   # noqa: E402
                            pick, propose)

GOLD = ROOT / "data" / "gold.jsonl"
PREDS = ROOT / "data" / "model_predictions.json"
ADAPTER = ROOT / "unsloth" / "lemmatizer_lora"
MAX_SEQ_LEN = 512
CHAT_TEMPLATE = "llama-3.2"

try:
    from llm import SYSTEM_PROMPT
except Exception:                                            # pragma: no cover
    SYSTEM_PROMPT = (
        "Anda adalah pelemma bahasa Melayu. Tukarkan ayat rojak kepada ayat "
        "berbentuk lema. Kembalikan ayat sahaja."
    )


def norm(w):
    return re.sub(r"[^\w]", "", w.lower())


# ---------------------------------------------------------------------------
# SYSTEMS
# ---------------------------------------------------------------------------

def predict_copy(row, lex, roots):
    return [e["t"] for e in row["lemmas"]]


def predict_malindo(row, lex, roots):
    """Plain dictionary lookup - what main.py/streamlit.py does today."""
    out = []
    for e in row["lemmas"]:
        tok = e["t"]
        lemma, _ = pick(lex.get(tok, []), tok)
        out.append(lemma or tok)
    return out


def predict_malindo_root(row, lex, roots):
    """MALINDO's root column - the dictionary's real morphological ability.

    Fairer to MALINDO than predict_malindo: the lemma column mostly echoes
    the surface form, so a lemma-column baseline makes the dictionary look
    far weaker than it is. This one strips affixes properly but still has
    no slang lexicon and no orthographic repair.
    """
    out = []
    for e in row["lemmas"]:
        tok = e["t"]
        root = roots.get(tok)
        if root:
            out.append(root)
        else:
            lemma, _ = pick(lex.get(tok, []), tok)
            out.append(lemma or tok)
    return out


def predict_rules(row, lex, roots):
    return [propose(e["t"], lex, roots)[0] for e in row["lemmas"]]


def _import_unsloth():
    """Import the real unsloth library, not the project folder of that name.

    The repo has a directory called `unsloth/`, so with the project root on
    sys.path (which it is whenever you run from there) `import unsloth`
    resolves to that folder as an empty namespace package and every attribute
    lookup fails with a misleading error. Drop the shadowing entries for the
    duration of the import.
    """
    shadow = {"", ".", str(ROOT)}
    saved = list(sys.path)
    sys.path[:] = [p for p in sys.path if p not in shadow]
    sys.modules.pop("unsloth", None)
    try:
        import unsloth
        if not hasattr(unsloth, "FastLanguageModel"):
            raise ImportError(
                "resolved unsloth to %r, which is not the library. "
                "Is the package installed in this interpreter?"
                % (getattr(unsloth, "__path__", None),))
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
        return FastLanguageModel, get_chat_template
    finally:
        sys.path[:] = saved


def generate_raw(rows):
    """Run the adapter and return its raw output text per sentence.

    Kept separate from alignment and cached to disk: a GPU pass costs
    minutes, and every change to the scoring logic would otherwise force
    another one.
    """
    FastLanguageModel, get_chat_template = _import_unsloth()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(ADAPTER), max_seq_length=MAX_SEQ_LEN,
        dtype=None, load_in_4bit=True)
    tokenizer = get_chat_template(tokenizer, chat_template=CHAT_TEMPLATE)
    FastLanguageModel.for_inference(model)

    out_texts = []
    for i, row in enumerate(rows, 1):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["text"]}]
        ids = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True,
            return_tensors="pt").to("cuda")
        out = model.generate(input_ids=ids, max_new_tokens=200, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
        out_texts.append(tokenizer.decode(out[0][ids.shape[1]:],
                                          skip_special_tokens=True).strip())
        if i % 25 == 0:
            print("   %d/%d" % (i, len(rows)))

    PREDS.write_text(json.dumps(
        {r["id"]: t for r, t in zip(rows, out_texts)},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print("   saved raw predictions -> %s" % PREDS.relative_to(ROOT))
    return out_texts


def align_all(rows, out_texts):
    """Align raw generations to gold tokens; report the format-failure rate."""
    preds, strict_ok = [], 0
    for row, text in zip(rows, out_texts):
        words = text.split()
        aligned = align_strict(words, row["lemmas"])
        if aligned is not None:
            strict_ok += 1
        else:
            aligned = align_lcs(words, row["lemmas"])
        preds.append(aligned)

    n = len(rows)
    print("   exact-length output : %d/%d (%.1f%%)"
          % (strict_ok, n, 100 * strict_ok / n))
    print("   length mismatch     : %d/%d (%.1f%%)  <- FORMAT-FAILURE RATE,"
          % (n - strict_ok, n, 100 * (n - strict_ok) / n))
    print("                         scored by LCS alignment, not zeroed")
    return preds


def align_strict(pred_words, lemmas):
    """Positional alignment. Only valid when the word counts agree exactly."""
    expected = sum(len(e["l"].split()) for e in lemmas)
    if len(pred_words) != expected:
        return None
    out, i = [], 0
    for e in lemmas:
        k = len(e["l"].split())
        out.append(" ".join(pred_words[i:i + k]))
        i += k
    return out


def align_lcs(pred_words, lemmas):
    """Longest-common-subsequence alignment, for when the lengths disagree.

    A generative model can drop or add a word anywhere in the sentence. With
    strict positional alignment one spurious word shifts everything after it
    and the whole sentence scores zero - which measures formatting, not
    lemmatisation. Matching on the longest common subsequence instead lets
    every token be judged on its own merits; only the genuinely unmatched
    ones are counted wrong.

    Returns a list the same length as `lemmas`: the matched prediction for
    each token, or None where nothing aligned.
    """
    gold_words, owner = [], []
    for i, e in enumerate(lemmas):
        for w in e["l"].split():
            gold_words.append(norm(w))
            owner.append(i)

    matcher = difflib.SequenceMatcher(
        None, gold_words, [norm(w) for w in pred_words], autojunk=False)

    matched = [False] * len(gold_words)
    got = [None] * len(lemmas)
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            gi, pi = block.a + k, block.b + k
            matched[gi] = True
            tok = owner[gi]
            got[tok] = (got[tok] or "") + (" " if got[tok] else "") + pred_words[pi]

    # A multi-word lemma only counts if every one of its words aligned.
    for i, e in enumerate(lemmas):
        need = [j for j, o in enumerate(owner) if o == i]
        if not all(matched[j] for j in need):
            got[i] = None
    return got


# ---------------------------------------------------------------------------
# SCORING
# ---------------------------------------------------------------------------

def score(rows, preds):
    """Returns overall, changed-subset, per-band and per-source accuracy."""
    s = {"all": [0, 0], "changed": [0, 0], "same": [0, 0],
         "band": defaultdict(lambda: [0, 0]), "src": defaultdict(lambda: [0, 0]),
         "exact": [0, 0]}

    for row, pred in zip(rows, preds):
        ok_sentence = True
        for e, p in zip(row["lemmas"], pred):
            hit = p is not None and norm(p) == norm(e["l"])
            changed = norm(e["t"]) != norm(e["l"])
            bucket = "changed" if changed else "same"
            for key in ("all", bucket):
                s[key][1] += 1
                s[key][0] += hit
            s["band"][row["band"]][1] += 1
            s["band"][row["band"]][0] += hit
            s["src"][e["src"]][1] += 1
            s["src"][e["src"]][0] += hit
            ok_sentence &= hit
        s["exact"][1] += 1
        s["exact"][0] += ok_sentence
    return s


def pct(pair):
    return 100 * pair[0] / pair[1] if pair[1] else 0.0


def main(with_model, reuse):
    if not GOLD.exists():
        sys.exit("No data/gold.jsonl. Run crawl/make_goldsheet.py --collect.")
    rows = [json.loads(l) for l in GOLD.open(encoding="utf-8") if l.strip()]

    print("loading MALINDO...")
    lex, roots = malindo_lemmas(), malindo_roots()

    n_tok = sum(len(r["lemmas"]) for r in rows)
    n_chg = sum(1 for r in rows for e in r["lemmas"]
                if norm(e["t"]) != norm(e["l"]))
    print("gold: %d sentences, %d tokens, %d need changing (%.0f%%)\n"
          % (len(rows), n_tok, n_chg, 100 * n_chg / n_tok))

    systems = [("copy", predict_copy), ("malindo", predict_malindo),
               ("dict-root", predict_malindo_root), ("rules", predict_rules)]
    results = {}
    for name, fn in systems:
        results[name] = score(rows, [fn(r, lex, roots) for r in rows])

    if with_model or reuse:
        texts = None
        if reuse:
            if not PREDS.exists():
                sys.exit("No %s yet. Run once with --model first." % PREDS.name)
            saved = json.loads(PREDS.read_text(encoding="utf-8"))
            missing = [r["id"] for r in rows if r["id"] not in saved]
            if missing:
                sys.exit("Cached predictions are stale - %d gold ids missing "
                         "(e.g. %s). Re-run with --model."
                         % (len(missing), missing[:3]))
            print("reusing cached predictions from %s\n" % PREDS.name)
            texts = [saved[r["id"]] for r in rows]
        elif not ADAPTER.exists():
            print("No adapter at %s - skipping finetune.\n" % ADAPTER)
        else:
            print("running the fine-tuned model...")
            texts = generate_raw(rows)

        if texts is not None:
            results["finetune"] = score(rows, align_all(rows, texts))
            print()

    names = list(results)
    print("=" * 66)
    print("%-10s %10s %10s %10s %10s" % ("system", "overall", "CHANGED",
                                         "unchanged", "sent-exact"))
    print("-" * 66)
    for n in names:
        s = results[n]
        print("%-10s %9.1f%% %9.1f%% %9.1f%% %9.1f%%"
              % (n, pct(s["all"]), pct(s["changed"]), pct(s["same"]),
                 pct(s["exact"])))
    print("=" * 66)
    print("`CHANGED` is the column that matters - tokens whose lemma differs")
    print("from the surface form. `copy` scores 0% there by definition.\n")

    print("%-10s %s" % ("by band", "".join("%12s" % b for b in
                                           ("low", "mid", "high"))))
    print("-" * 48)
    for n in names:
        print("%-10s %s" % (n, "".join(
            "%11.1f%%" % pct(results[n]["band"][b])
            for b in ("low", "mid", "high"))))

    srcs = sorted(results[names[0]]["src"],
                  key=lambda k: -results[names[0]]["src"][k][1])
    print("\n%-12s %6s %s" % ("by gold src", "n",
                              "".join("%11s" % n for n in names)))
    print("-" * (20 + 11 * len(names)))
    for src in srcs:
        n_src = results[names[0]]["src"][src][1]
        print("%-12s %6d %s" % (src, n_src, "".join(
            "%10.1f%%" % pct(results[n]["src"][src]) for n in names)))

    human = results[names[0]]["src"].get("human", [0, 0])[1]
    print()
    print("=" * 66)
    print("HONEST SUBSET - the %d tokens an annotator had to override" % human)
    print("=" * 66)
    for n in names:
        print("   %-12s %5.1f%%" % (n, pct(results[n]["src"]["human"])))
    print()
    print("Read THIS table, not the headline one. The gold labels were")
    print("pre-filled by `rules`, so on the other %d tokens the annotator"
          % (sum(len(r["lemmas"]) for r in rows) - human))
    print("mostly just accepted the rule output - and `rules` then scores")
    print("itself correct there. Only the rows a human overrode are")
    print("independent of the rule system, and they are the only place a")
    print("fine-tune can prove it adds anything a dictionary cannot.")
    print()
    print("Caveat for the report: n=%d is small, so this figure is noisy."
          % human)
    print("Label more gold sentences if the final margin looks close.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="store_true",
                   help="also evaluate the fine-tuned adapter (needs GPU)")
    p.add_argument("--reuse", action="store_true",
                   help="score cached predictions instead of re-running the "
                        "model (no GPU needed)")
    a = p.parse_args()
    main(a.model, a.reuse)
