import argparse
import difflib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "crawl"))
sys.path.insert(0, str(ROOT / "unsloth"))
sys.path.insert(0, str(ROOT))

GOLD = ROOT / "data" / "gold.jsonl"
SILVER = ROOT / "data" / "silver.jsonl"
MANIFEST = ROOT / "data" / "split_manifest.json"
VAL = ROOT / "data" / "val.jsonl"
BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
BASE_MODEL_ALT = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
BASE_MODELS = (BASE_MODEL, BASE_MODEL_ALT)

SPLIT = "gold"
MAX_SEQ_LEN = 1024

CHAT_TEMPLATES = (("qwen", "qwen-2.5"), ("llama", "llama-3.2"))
DEFAULT_TEMPLATE = "llama-3.2"

import skill


def _tag(ref):
    name = getattr(ref, "name", None) or str(ref).rstrip("/").split("/")[-1]
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def label_for(ref):
    name = _tag(ref)
    for fam in ("llama", "qwen"):
        if name.startswith("lemmatizer_lora_") and fam in name:
            return "%s-ft" % fam
    return name


def preds_path(adapter, split=None):
    return ROOT / "data" / ("model_predictions_%s_%s.json"
                            % (split or SPLIT, _tag(adapter)))


def template_for(ref):
    low = str(ref).lower()
    for needle, template in CHAT_TEMPLATES:
        if needle in low:
            return template
    print("[eval] cannot tell which family %r belongs to - falling back to %s."
          % (str(ref), DEFAULT_TEMPLATE))
    return DEFAULT_TEMPLATE


def prompt_for(ref):
    if str(ref) in BASE_MODELS:
        return skill.SYSTEM_PROMPT
    prompt, _version = skill.prompt_for(str(ref))
    return prompt


def norm(w):
    return re.sub(r"[^\w]", "", w.lower())


def _import_unsloth():
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
    except Exception as e:
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
    preds = []
    for row, text in zip(rows, out_texts):
        parsed = skill.parse_response(text)
        lemmas = [s["lemma"] for s in parsed["steps"]]

        if parsed["ok"] and len(lemmas) == len(row["lemmas"]):
            preds.append(lemmas)
        elif parsed["steps"]:
            preds.append(align_lcs(lemmas, row["lemmas"]))
        else:
            preds.append(align_lcs(text.split(), row["lemmas"]))
    return preds


def lemma_sentences(out_texts):
    out = []
    for text in out_texts:
        parsed = skill.parse_response(text)
        out.append(parsed["sentence"] or text)
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

    for i, e in enumerate(lemmas):
        need = [j for j, o in enumerate(owner) if o == i]
        if not all(matched[j] for j in need):
            got[i] = None
    return got


def edit_distance(a, b):
    """Levenshtein distance between two sequences."""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def sequence_metrics(rows, preds, raw_texts=None):
    """Corpus-level WER and CER of the lemma sentence against the gold one."""
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


def score(rows, preds):
    s = {"all": [0, 0], "cm": {"tp": 0, "fp": 0, "fn": 0, "tn": 0}}

    for row, pred in zip(rows, preds):
        for e, p in zip(row["lemmas"], pred):
            hit = p is not None and norm(p) == norm(e["l"])
            changed = norm(e["t"]) != norm(e["l"])
            acted = p is not None and norm(p) != norm(e["t"])

            if changed and acted and hit:
                s["cm"]["tp"] += 1
            elif changed and not acted:
                s["cm"]["fn"] += 1
            elif changed and acted:
                s["cm"]["fp"] += 1
                s["cm"]["fn"] += 1
            elif acted:
                s["cm"]["fp"] += 1
            else:
                s["cm"]["tn"] += 1

            s["all"][1] += 1
            s["all"][0] += hit
    return s


def pct(pair):
    return 100 * pair[0] / pair[1] if pair[1] else 0.0


def prf(cm):
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


def run_neural(name, rows, ref, reuse, results):
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
        weights = ref / "adapter_model.safetensors" if hasattr(ref, "exists") else None
        if weights is not None and weights.exists() and \
                weights.stat().st_mtime > cache.stat().st_mtime:
            print("STALE: %s was written before %s was retrained.\n"
                  "       Re-run with --model; scoring it would report the "
                  "old adapter.\n" % (cache.name, ref.name))
            return
        texts = [saved[r["id"]] for r in rows]
    else:
        if hasattr(ref, "exists") and not ref.exists():
            print("No adapter at %s - skipping %s.\n" % (ref, name))
            return
        print("running %s ...\n" % name)
        texts = generate_raw(rows, ref, prompt_for(ref))

    preds = align_all(rows, texts)
    results[name] = score(rows, preds)
    results[name]["wer"], results[name]["cer"] = sequence_metrics(
        rows, preds, raw_texts=lemma_sentences(texts))


def report(names, results, n_tok):
    print("=" * 78)
    print("%-12s %10s %10s %10s %8s %9s %9s"
          % ("system", "accuracy", "precision", "recall", "F1", "WER", "CER"))
    print("-" * 78)
    for n in names:
        r = results[n]
        pr, rc, f1 = prf(r["cm"])
        print("%-12s %9.1f%% %9.1f%% %9.1f%% %7.1f%% %9.4f %9.4f"
              % (n, pct(r["all"]), pr, rc, f1,
                 r.get("wer", 0.0), r.get("cer", 0.0)))
    print("=" * 78)
    print()

    print("=" * 78)
    print("%-12s %8s %8s %8s %8s %9s %8s %8s"
          % ("system", "TP", "FP", "FN", "TN", "precision", "recall", "F1"))
    print("-" * 78)
    for n in names:
        cm = results[n]["cm"]
        pr, rc, f1 = prf(cm)
        print("%-12s %8d %8d %8d %8d %8.1f%% %7.1f%% %7.1f%%"
              % (n, cm["tp"], cm["fp"], cm["fn"], cm["tn"], pr, rc, f1))
    print("=" * 78)

    over = [(n, sum(results[n]["cm"].values()) - n_tok) for n in names
            if sum(results[n]["cm"].values()) > n_tok]
    if over:
        print("cells exceed the %d tokens (%s):"
              % (n_tok, ", ".join("%s +%d" % x for x in over)))
        print("a token that needed a change and got the wrong one counts as "
              "both FP and FN.")
    print()


def main(with_model, reuse, adapters, with_base=False, on_dev=False,
         reuse_base=False, on_val=False, with_base_alt=False):
    global SPLIT
    SPLIT = "dev" if on_dev else ("val" if on_val else "gold")

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

    n_tok = sum(len(r["lemmas"]) for r in rows)
    n_chg = sum(1 for r in rows for e in r["lemmas"]
                if norm(e["t"]) != norm(e["l"]))
    print("%s: %d sentences, %d tokens, %d need changing (%.0f%%)\n"
          % (label, len(rows), n_tok, n_chg, 100 * n_chg / n_tok))

    results = {}
    for name, ref in (("base-llama", BASE_MODEL), ("base-qwen", BASE_MODEL_ALT)):
        wanted = (with_base and ref == BASE_MODEL) or \
                 (with_base_alt and ref == BASE_MODEL_ALT) or reuse_base or \
                 (reuse and preds_path(ref).exists())
        if wanted:
            run_neural(name, rows, ref, reuse or reuse_base, results)

    for ad in adapters:
        if with_model or (reuse and preds_path(ad).exists()):
            run_neural(label_for(ad), rows, ad, reuse, results)

    if not results:
        sys.exit("Nothing to score. Add --base / --base-qwen / --model, or "
                 "--reuse to score cached predictions.")
    report(list(results), results, n_tok)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="store_true",
                   help="also evaluate the fine-tuned adapter (needs GPU)")
    p.add_argument("--base-qwen", action="store_true",
                   help="also evaluate the UNFINE-TUNED Qwen base model")
    p.add_argument("--adapter", action="append", default=None,
                   help="adapter folder under unsloth/ "
                        "(default lemmatizer_lora_llama)")
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
    p.add_argument("--val", action="store_true",
                   help="score the human-labelled VALIDATION split "
                        "(data/val.jsonl) - this is where model selection "
                        "happens, not on gold")
    p.add_argument("--dev", action="store_true",
                   help="score the dev split instead of the gold test set")
    a = p.parse_args()
    if a.val and a.dev:
        p.error("--val and --dev name different splits; pick one.")
    names = a.adapter or ["lemmatizer_lora_llama"]
    main(a.model, a.reuse, [ROOT / "unsloth" / n for n in names],
         a.base, a.dev, reuse_base=a.reuse_base, on_val=a.val,
         with_base_alt=a.base_qwen)
