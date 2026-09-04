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
sys.path.insert(0, str(ROOT))

from make_goldsheet import (malindo_lemmas, malindo_roots,   # noqa: E402
                            pick, propose)

GOLD = ROOT / "data" / "gold.jsonl"
SILVER = ROOT / "data" / "silver.jsonl"
MANIFEST = ROOT / "data" / "split_manifest.json"
VAL = ROOT / "data" / "val.jsonl"
BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
BASE_MODEL_ALT = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
BASE_MODELS = (BASE_MODEL, BASE_MODEL_ALT)
ADAPTER = ROOT / "unsloth" / "lemmatizer_lora_yt"   # overridden by --adapter


def label_for(ref):
    name = _tag(ref)
    for fam in ("llama", "qwen"):
        if name.startswith("lemmatizer_lora_") and fam in name:
            return "%s-ft" % fam
    return name


def preds_path(adapter):
    return ROOT / "data" / ("model_predictions_%s.json" % _tag(adapter))


def _tag(ref):
    name = getattr(ref, "name", None) or str(ref).rstrip("/").split("/")[-1]
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)

MAX_SEQ_LEN = 1024
CHAT_TEMPLATES = (("qwen", "qwen-2.5"), ("llama", "llama-3.2"))
DEFAULT_TEMPLATE = "llama-3.2"


def template_for(ref):
    """The chat template this model family needs."""
    low = str(ref).lower()
    for needle, template in CHAT_TEMPLATES:
        if needle in low:
            return template
    print("[eval] cannot tell which family %r belongs to - falling back to %s. "
          "If its scores look inexplicably bad, this is why."
          % (str(ref), DEFAULT_TEMPLATE))
    return DEFAULT_TEMPLATE


import skill                                               


def prompt_for(ref):
    if str(ref) in BASE_MODELS:
        # Never fine-tuned, so it has no stamp to read and no opinion to ask.
        return skill.SYSTEM_PROMPT
    prompt, _version = skill.prompt_for(str(ref))
    return prompt


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


def _free_gpu(note=""):
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info()
            print("   GPU %s: %.2f of %.2f GB free"
                  % (note or "after unload", free / 1e9, total / 1e9))
    except Exception as e:                      # CPU-only box, or no driver
        print("   (could not query the GPU: %s)" % e)


def generate_raw(rows, adapter, system_prompt=None):
    FastLanguageModel, get_chat_template = _import_unsloth()
    if system_prompt is None:
        system_prompt = prompt_for(adapter)

    _free_gpu("before load")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter), max_seq_length=MAX_SEQ_LEN,
        dtype=None, load_in_4bit=True)
    tokenizer = get_chat_template(tokenizer, chat_template=template_for(adapter))
    FastLanguageModel.for_inference(model)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.max_length = None

    out_texts = []
    for i, row in enumerate(rows, 1):
        msgs = [{"role": "system", "content": system_prompt},
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

    cache = preds_path(adapter)
    cache.write_text(json.dumps(
        {r["id"]: t for r, t in zip(rows, out_texts)},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print("   saved raw predictions -> %s" % cache.relative_to(ROOT))

    del model, tokenizer
    _free_gpu()
    return out_texts


def align_all(rows, out_texts):
    preds, strict_ok, unparseable = [], 0, 0
    for row, text in zip(rows, out_texts):
        parsed = skill.parse_response(text)
        lemmas = [s["lemma"] for s in parsed["steps"]]

        if parsed["ok"] and len(lemmas) == len(row["lemmas"]):
            # One line per token, in order: the format did its job.
            preds.append(lemmas)
            strict_ok += 1
            continue

        if parsed["steps"]:
            # A derivation, but the wrong length -- lines dropped, invented or
            # merged. Salvage by aligning the LEMMA COLUMN, which is still far
            # better evidence than the raw tag soup.
            preds.append(align_lcs(lemmas, row["lemmas"]))
        else:
            # No parseable derivation at all. This is the real format failure,
            # and it is what an un-fine-tuned base model does.
            unparseable += 1
            preds.append(align_lcs(text.split(), row["lemmas"]))

    n = len(rows)
    print("   one line per token  : %d/%d (%.1f%%)"
          % (strict_ok, n, 100 * strict_ok / n))
    print("   parsed, wrong length: %d/%d (%.1f%%)  <- salvaged by aligning"
          % (n - strict_ok - unparseable, n,
             100 * (n - strict_ok - unparseable) / n))
    print("   no derivation at all: %d/%d (%.1f%%)  <- FORMAT-FAILURE RATE"
          % (unparseable, n, 100 * unparseable / n))
    return preds, 100.0 * unparseable / n


def lemma_sentences(out_texts):
    out = []
    for text in out_texts:
        parsed = skill.parse_response(text)
        out.append(parsed["sentence"] or text)
    return out


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



def edit_distance(a, b):
    """Levenshtein distance between two sequences, O(len(a) x len(b))."""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1,          # deletion
                           cur[j - 1] + 1,       # insertion
                           prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def sequence_metrics(rows, preds, raw_texts=None):
    w_err = w_len = c_err = c_len = 0
    for i, (row, pred) in enumerate(zip(rows, preds)):
        ref = " ".join(norm(w) for e in row["lemmas"]
                       for w in e["l"].split() if norm(w))
        if raw_texts is not None:
            hyp = " ".join(norm(w) for w in raw_texts[i].split() if norm(w))
        else:
            hyp = " ".join(norm(p) for p in pred if p and norm(p))

        rw, hw = ref.split(), hyp.split()
        w_err += edit_distance(rw, hw)
        w_len += len(rw)
        c_err += edit_distance(ref.replace(" ", ""), hyp.replace(" ", ""))
        c_len += len(ref.replace(" ", ""))

    return (w_err / w_len if w_len else 0.0,
            c_err / c_len if c_len else 0.0)


def mcnemar(a, b):
    from math import comb
    n01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    n10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return n01, n10, min(p, 1.0)


# ---------------------------------------------------------------------------
# SCORING
# ---------------------------------------------------------------------------

def score(rows, preds):
    """Returns overall, changed-subset, per-band and per-source accuracy."""
    s = {"all": [0, 0], "changed": [0, 0], "same": [0, 0],
         "band": defaultdict(lambda: [0, 0]), "src": defaultdict(lambda: [0, 0]),
         "exact": [0, 0],
         # tp/fp/fn/tn for the decision to CHANGE a token, plus the per-token
         # correctness vector McNemar needs.
         "cm": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
         "vec": [], "vec_human": []}

    for row, pred in zip(rows, preds):
        ok_sentence = True
        for e, p in zip(row["lemmas"], pred):
            hit = p is not None and norm(p) == norm(e["l"])
            changed = norm(e["t"]) != norm(e["l"])
            bucket = "changed" if changed else "same"

            acted = p is not None and norm(p) != norm(e["t"])
            if changed and acted and hit:
                s["cm"]["tp"] += 1            # changed it, and correctly
            elif changed and not acted:
                s["cm"]["fn"] += 1            # left one that needed changing
            elif changed and acted:
                s["cm"]["fp"] += 1            # changed it, but wrongly
                s["cm"]["fn"] += 1            # and still missed the right one
            elif acted:
                s["cm"]["fp"] += 1            # OVER-CORRECTION
            else:
                s["cm"]["tn"] += 1
            s["vec"].append(int(hit))
            if e["src"] == "human":
                s["vec_human"].append(int(hit))
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


def prf(cm):
    """(precision, recall, F1) from the confusion counts, as percentages."""
    tp, fp, fn = cm["tp"], cm["fp"], cm["fn"]
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return 100 * p, 100 * r, 100 * f


def load_dev_rows():
    if not MANIFEST.exists():
        sys.exit("No data/split_manifest.json. Run "
                 "`python crawl/build_silver.py --resplit` first.")
    dev_ids = set(json.loads(MANIFEST.read_text(encoding="utf-8"))["dev_ids"])
    rows = [json.loads(l) for l in SILVER.open(encoding="utf-8") if l.strip()]
    return [r for r in rows if r["id"] in dev_ids]


def run_neural(name, rows, ref, reuse, results, fmt_fail):
    """Score one neural system, from cache or by generating."""
    cache = preds_path(ref)
    if reuse:
        if not cache.exists():
            print("No %s yet - skipping %s.\n" % (cache.name, name))
            return
        saved = json.loads(cache.read_text(encoding="utf-8"))
        missing = [r["id"] for r in rows if r["id"] not in saved]
        if missing:
            print("Cached predictions for %s are stale - %d ids missing "
                  "(e.g. %s). Re-run with --model.\n"
                  % (name, len(missing), missing[:3]))
            return
        # Same ids is not the same model. The cache is keyed on the adapter
        # NAME, so retraining in place leaves a complete-looking file full of
        # the old adapter's answers - which would be scored and reported as
        # the new one's without a word of warning.
        weights = ref / "adapter_model.safetensors" if hasattr(ref, "exists") else None
        if weights is not None and weights.exists() and \
                weights.stat().st_mtime > cache.stat().st_mtime:
            print("STALE: %s was written before %s was retrained.\n"
                  "       Re-run with --model; scoring it would report the "
                  "old adapter.\n" % (cache.name, ref.name))
            return
        print("reusing cached predictions from %s\n" % cache.name)
        texts = [saved[r["id"]] for r in rows]
    else:
        if hasattr(ref, "exists") and not ref.exists():
            print("No adapter at %s - skipping %s.\n" % (ref, name))
            return
        print("running %s ...\n" % name)
        texts = generate_raw(rows, ref, prompt_for(ref))

    preds, fail = align_all(rows, texts)
    results[name] = score(rows, preds)
    results[name]["preds"] = preds
    results[name]["wer"], results[name]["cer"] = sequence_metrics(
        rows, preds, raw_texts=lemma_sentences(texts))
    fmt_fail[name] = fail
    print()


def error_table(rows, results, a_name="rules", b_name=None):
    if a_name not in results or b_name not in results:
        print("\nNeed both %s and %s scored - add --model (and --base).\n"
              % (a_name, b_name))
        return
    A, B = results[a_name].get("preds"), results[b_name].get("preds")
    if A is None or B is None:
        return

    def cell(x):
        return "-" if x is None else str(x)

    a_only, b_only, both_bad = [], [], 0
    for r, pa, pb in zip(rows, A, B):
        for i, e in enumerate(r["lemmas"]):
            g = norm(e["l"])
            va = norm(cell(pa[i]))
            vb = norm(cell(pb[i]))
            rec = (e["t"], e["l"], cell(pa[i]), cell(pb[i]),
                   e.get("src", "?"),
                   " ".join(x["t"] for x in r["lemmas"]))
            if va == g and vb != g:
                a_only.append(rec)
            elif vb == g and va != g:
                b_only.append(rec)
            elif va != g and vb != g:
                both_bad += 1

    total = sum(len(r["lemmas"]) for r in rows)
    print()
    print("=" * 78)
    print("ERROR ANALYSIS - %s vs %s, every disagreement" % (a_name, b_name))
    print("=" * 78)
    print("only %-10s correct : %4d" % (b_name, len(b_only)))
    print("only %-10s correct : %4d" % (a_name, len(a_only)))
    print("both wrong              : %4d   <- the shared blind spot" % both_bad)
    print("both correct            : %4d"
          % (total - len(a_only) - len(b_only) - both_bad))
    print("total tokens            : %4d" % total)

    for title, recs, other in ((b_name, b_only, a_name),
                               (a_name, a_only, b_name)):
        print()
        print("-" * 78)
        print("TOKENS ONLY `%s` GOT RIGHT (%d)" % (title, len(recs)))
        print("-" * 78)
        print("  %-14s %-14s %-14s %s" % ("token", "gold", "%s said" % other,
                                          "src"))
        for tok, gold, pa, pb, src, ctx in recs:
            wrong = pa if title == b_name else pb
            print("  %-14s %-14s %-14s %s" % (tok, gold, wrong, src))
            print("       in: %s" % ctx[:92])

    # The other claim worth being able to defend: how much of this corpus
    # actually needs context at all?
    from collections import defaultdict
    seen = defaultdict(set)
    for r in rows:
        for e in r["lemmas"]:
            seen[norm(e["t"])].add(norm(e["l"]))
    amb = {t: v for t, v in seen.items() if len(v) > 1}
    n_amb = sum(1 for r in rows for e in r["lemmas"] if norm(e["t"]) in amb)
    print()
    print("=" * 78)
    print("HOW MUCH OF THIS CORPUS NEEDS CONTEXT?")
    print("=" * 78)
    print("A per-token system like `rules` cannot handle a surface form that")
    print("needs different lemmas in different sentences. Counting them:")
    print()
    print("  distinct surface forms in gold          : %d" % len(seen))
    print("  forms needing >1 lemma across sentences : %d" % len(amb))
    print("  token occurrences of those forms        : %d of %d (%.1f%%)"
          % (n_amb, total, 100.0 * n_amb / total))
    for t, v in sorted(amb.items()):
        print("     %-14s -> %s" % (t, " / ".join(sorted(v))))
    print()
    print("So context-dependence is a real structural limit of any per-token")
    print("rule system, and this corpus contains almost none of it. Quote it")
    print("as a mechanism, not as a measured advantage.")


def main(with_model, reuse, adapters, with_base=False, on_dev=False,
         reuse_base=False, show_errors=False, on_val=False,
         with_base_alt=False, models_only=False):
    if on_dev:
        rows = load_dev_rows()
        label = "dev"
    elif on_val:
        if not VAL.exists():
            sys.exit("No data/val.jsonl. Collect it first:\n"
                     "  python crawl/make_goldsheet.py --collect "
                     "--pool data/val_pool.jsonl --sheet data/val_sheet.xlsx "
                     "--out data/val.jsonl")
        rows = [json.loads(l) for l in VAL.open(encoding="utf-8") if l.strip()]
        label = "val (model selection)"
    else:
        if not GOLD.exists():
            sys.exit("No data/gold.jsonl. Run crawl/make_goldsheet.py --collect.")
        rows = [json.loads(l) for l in GOLD.open(encoding="utf-8") if l.strip()]
        label = "gold TEST"

    print("loading MALINDO...")
    lex, roots = malindo_lemmas(), malindo_roots()

    n_tok = sum(len(r["lemmas"]) for r in rows)
    n_chg = sum(1 for r in rows for e in r["lemmas"]
                if norm(e["t"]) != norm(e["l"]))
    print("%s: %d sentences, %d tokens, %d need changing (%.0f%%)\n"
          % (label, len(rows), n_tok, n_chg, 100 * n_chg / n_tok))
    if on_dev:
        print("NOTE: dev is silver-labelled, so `rules` scores ~100% here BY")
        print("      CONSTRUCTION - the silver labels are its own output. The")
        print("      only meaningful use of this split is monitoring the")
        print("      fine-tune during development. Every reported comparison")
        print("      belongs on the gold test set.\n")

    systems = [("copy", predict_copy), ("malindo", predict_malindo),
               ("dict-root", predict_malindo_root), ("rules", predict_rules)]
    results, fmt_fail = {}, {}
    for name, fn in systems:
        preds = [fn(r, lex, roots) for r in rows]
        results[name] = score(rows, preds)
        results[name]["preds"] = preds
        results[name]["wer"], results[name]["cer"] = sequence_metrics(rows, preds)

    for label, ref in (("base-llama", BASE_MODEL), ("base-qwen", BASE_MODEL_ALT)):
        cached = preds_path(ref).exists()
        wanted = (with_base and ref == BASE_MODEL) or \
                 (with_base_alt and ref == BASE_MODEL_ALT) or reuse_base
        if wanted or (reuse and cached):
            run_neural(label, rows, ref, (reuse or reuse_base) and cached,
                       results, fmt_fail)
    if reuse and not any(preds_path(r).exists() for r in BASE_MODELS):
        print("no base-model cache yet - run once with --base / --base-qwen to "
              "add the before-fine-tuning rows.\n")

    for ad in adapters:
        if with_model or (reuse and preds_path(ad).exists()):
            run_neural(label_for(ad), rows, ad, reuse, results, fmt_fail)

    names = list(results)
    if models_only:
        keep = [n for n in names if n.startswith("base-") or n.endswith("-ft")]
        if keep:
            names = keep
            print("--models-only: showing the %d language models; the "
                  "non-neural baselines are omitted.\n" % len(names))
    print("=" * 80)
    print("%-12s %10s %10s %10s %11s %12s"
          % ("system", "overall", "CHANGED", "unchanged", "sent-exact",
             "format-fail"))
    print("-" * 80)
    for n in names:
        r = results[n]
        ff = ("%11.1f%%" % fmt_fail[n]) if n in fmt_fail else "%11s" % "n/a"
        print("%-12s %9.1f%% %9.1f%% %9.1f%% %10.1f%% %s"
              % (n, pct(r["all"]), pct(r["changed"]), pct(r["same"]),
                 pct(r["exact"]), ff))
    print("=" * 80)
    print("`CHANGED` is the column that matters - tokens whose lemma differs")
    print("from the surface form. `copy` scores 0% there by definition.\n")

    # ---- classification view of the same predictions ----------------------
    print("=" * 80)
    print("%-12s %8s %8s %8s %8s   %8s %8s %8s"
          % ("system", "TP", "FP", "FN", "TN",
             "precision", "recall", "F1"))
    print("-" * 80)
    for n in names:
        cm = results[n]["cm"]
        p, r, f = prf(cm)
        print("%-12s %8d %8d %8d %8d   %8.1f%% %7.1f%% %7.1f%%"
              % (n, cm["tp"], cm["fp"], cm["fn"], cm["tn"], p, r, f))
    print("=" * 80)
    print("Framed as the decision to CHANGE a token. RECALL is the same number")
    print("as the CHANGED column above. PRECISION is the one accuracy hides:")
    print("FP counts OVER-CORRECTION - a token that should have been left")
    print("alone but was rewritten. F1 is the honest single number.\n")

    # ---- sequence-level error rates ---------------------------------------
    print("%-12s %10s %10s" % ("system", "WER", "CER"))
    print("-" * 33)
    for n in names:
        print("%-12s %9.4f %9.4f"
              % (n, results[n].get("wer", 0.0), results[n].get("cer", 0.0)))
    print("-" * 33)
    print("Corpus-level word and character error rate of the produced lemma")
    print("sentence against the gold one. CER gives partial credit where token")
    print("accuracy scores a flat zero: `tengo` for `tengok` is 1 character")
    print("wrong, not a whole token wrong.\n")

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

    # ---- is any of it significant? ----------------------------------------
    if len(names) > 1:
        ref = "rules" if "rules" in names else names[0]
        print()
        print("=" * 78)
        print("McNEMAR vs `%s` - are the differences larger than chance?" % ref)
        print("=" * 78)
        print("%-12s %-22s %8s %8s %10s  %s"
              % ("system", "subset", "it wins", ref + " wins", "p", ""))
        print("-" * 78)
        for n in names:
            if n == ref:
                continue
            for label, key in (("all %d tokens" % len(results[n]["vec"]), "vec"),
                               ("%d honest tokens" % len(results[n]["vec_human"]),
                                "vec_human")):
                a, b = results[ref][key], results[n][key]
                if not a:
                    continue
                n01, n10, p = mcnemar(a, b)
                verdict = "SIGNIFICANT" if p < 0.05 else "not significant"
                shown = "<1e-300" if p == 0.0 else "%.4g" % p
                print("%-12s %-22s %8d %8d %10s  %s"
                      % (n, label, n01, n10, shown, verdict))
        print("-" * 78)
        print("Only the tokens the two systems DISAGREE on carry information;")
        print("the counts above are exactly those. p < 0.05 means the gap is")
        print("unlikely to be chance - it does not mean the gap is large.")

    fts = [n for n in names if n.endswith("-ft")]
    if len(fts) >= 2:
        print()
        print("=" * 78)
        print("HEAD TO HEAD - which fine-tune to select")
        print("=" * 78)
        for i in range(len(fts)):
            for j in range(i + 1, len(fts)):
                a_name, b_name = fts[i], fts[j]
                for label, key in (("all tokens", "vec"),
                                   ("honest subset", "vec_human")):
                    a, b = results[a_name][key], results[b_name][key]
                    if not a:
                        continue
                    n01, n10, p = mcnemar(a, b)
                    shown = "<1e-300" if p == 0.0 else "%.4g" % p
                    print("\n%s vs %s  (%s, n=%d)"
                          % (a_name, b_name, label, len(a)))
                    print("  %s only correct : %d" % (b_name, n01))
                    print("  %s only correct : %d" % (a_name, n10))
                    print("  p = %s  ->  %s" % (
                        shown,
                        "a real difference" if p < 0.05 else
                        "TIED: the gap is within noise, so pick on the "
                        "pre-declared tie-break"))
        print()
        print("A tie here is the expected outcome, not a failure. Two 3B models")
        print("trained on identical data usually land within noise of each")
        print("other; the honest report says so and selects on the criterion")
        print("declared before the numbers were seen.")

    if show_errors:
        # Whichever fine-tune is present; with both, the first one asked for.
        fts = [n for n in results if n.endswith("-ft")]
        if fts:
            error_table(rows, results, b_name=fts[0])
        else:
            print("--errors needs a fine-tuned adapter in the run.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="store_true",
                   help="also evaluate the fine-tuned adapter (needs GPU)")
    p.add_argument("--models-only", action="store_true",
                   help="report only the language models, omitting copy / "
                        "MALINDO / rules from every table")
    p.add_argument("--base-qwen", action="store_true",
                   help="also evaluate the UNFINE-TUNED Qwen base model")
    p.add_argument("--adapter", action="append", default=None,
                   help="adapter folder under unsloth/ "
                        "(default lemmatizer_lora_yt)")
    p.add_argument("--reuse", action="store_true",
                   help="score cached predictions instead of re-running the "
                        "model (no GPU needed)")
    p.add_argument("--base", action="store_true",
                   help="also evaluate the UNFINE-TUNED base model "
                        "(%s) with the same prompt the fine-tune was trained "
                        "on" % BASE_MODEL)
    p.add_argument("--reuse-base", action="store_true",
                   help="score the base model from its cache while still "
                        "generating the fine-tune - one model on the GPU at a "
                        "time, and no second base pass")
    p.add_argument("--errors", action="store_true",
                   help="print every token rules and the fine-tune disagree "
                        "on, with the gold answer - reproduces the error "
                        "analysis table")
    p.add_argument("--val", action="store_true",
                   help="score the human-labelled VALIDATION split "
                        "(data/val.jsonl) - this is where model selection "
                        "happens, not on gold")
    p.add_argument("--dev", action="store_true",
                   help="score the 300-sentence dev split instead of the gold "
                        "test set. For development only - the reported "
                        "comparison belongs on gold.")
    a = p.parse_args()
    if a.val and a.dev:
        p.error("--val and --dev name different splits; pick one.")
    names = a.adapter or ["lemmatizer_lora_llama"]
    main(a.model, a.reuse, [ROOT / "unsloth" / n for n in names],
         a.base, a.dev, reuse_base=a.reuse_base, show_errors=a.errors,
         on_val=a.val, with_base_alt=a.base_qwen,
         models_only=a.models_only)
