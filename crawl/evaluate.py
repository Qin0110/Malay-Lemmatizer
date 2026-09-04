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

    finetune     the QLoRA adapter in unsloth/lemmatizer_lora_yt.

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
# skill.py and preprocess.py live at the project root: they are the contract
# between the data pipeline and the model, so neither half owns them.
sys.path.insert(0, str(ROOT))

from make_goldsheet import (malindo_lemmas, malindo_roots,   # noqa: E402
                            pick, propose)

GOLD = ROOT / "data" / "gold.jsonl"
SILVER = ROOT / "data" / "silver.jsonl"
MANIFEST = ROOT / "data" / "split_manifest.json"
VAL = ROOT / "data" / "val.jsonl"
# The two un-fine-tuned baselines. Matched on purpose: both are ~3B
# instruction-tuned decoders in the same 4-bit Unsloth packaging, so the only
# thing that differs between the two families is the pretraining, not the size
# or the quantisation. Qwen carries markedly more Malay in its pretraining mix
# than Llama 3.2 does, which is what makes the comparison worth running rather
# than a foregone conclusion.
BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
BASE_MODEL_ALT = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
BASE_MODELS = (BASE_MODEL, BASE_MODEL_ALT)
ADAPTER = ROOT / "unsloth" / "lemmatizer_lora_yt"   # overridden by --adapter


def label_for(ref):
    """A short row label for the results table.

    The cache is named after the adapter folder, which is deliberately long and
    unambiguous; a results table wants "llama-ft", not
    "lemmatizer_lora_llama". Anything unrecognised keeps its folder name rather
    than being forced into a scheme it does not fit.
    """
    name = _tag(ref)
    for fam in ("llama", "qwen"):
        if name.startswith("lemmatizer_lora_") and fam in name:
            return "%s-ft" % fam
    return name


def preds_path(adapter):
    """One cache per adapter, named after it.

    A single shared cache silently went stale once already: predictions
    generated from one adapter were still being scored after a different
    adapter replaced it on disk. Naming the cache after the adapter makes
    that impossible - a different model simply has no cache yet.
    """
    return ROOT / "data" / ("model_predictions_%s.json" % _tag(adapter))


def _tag(ref):
    """A filesystem-safe name for whatever we are about to run.

    The base model is a Hugging Face id with a slash in it, and it must not
    land in the same cache file as an adapter -- reusing one system's
    predictions to score another is precisely the mistake the per-adapter
    cache exists to prevent.
    """
    name = getattr(ref, "name", None) or str(ref).rstrip("/").split("/")[-1]
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
# The prompt alone is ~466 tokens at the median under the Llama tokeniser (the
# v2 reasoning system prompt is long), and generation adds up to 200 more. At
# the old value of 512 -- a leftover from v1, when the prompt was one line and
# the answer a bare sentence -- the model ran out of window mid-derivation and
# unsloth truncated it. Sized to the measured worst case with headroom.
MAX_SEQ_LEN = 1024

# Which chat template each family needs. This was a single hardcoded
# "llama-3.2" applied to every model, which meant Qwen was wrapped in Llama's
# special tokens. That does not raise: the model answers fluently in the wrong
# format, so it simply scores badly and looks like a weak model rather than a
# misconfigured one. Same failure mode the loss-mask markers have in train.py,
# and the reason both are derived per family rather than assumed.
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

# Ask the adapter which prompt it was trained with, exactly as chat.py does.
#
# This used to be `from llm import SYSTEM_PROMPT`, and that was a live bug.
# llm.py selects v1 or v2 from the USE_REASONING environment variable, which
# describes a TRAINING run rather than the model being scored -- so with that
# variable unset (default "1") every evaluation of the v1 lemmatizer_lora_yt
# was handing it the v2 reasoning prompt. The failure is silent: the model
# still answers fluently, the answer is just the wrong format, and the score
# comes out low for a reason that has nothing to do with the model.
import skill                                                 # noqa: E402


def prompt_for(ref):
    """The system prompt this model expects.

    This project has ONE skill version, so every system in the comparison --
    both base models and both fine-tunes -- is served the identical prompt.
    That is the point: the four-way comparison is about the weights, and
    varying the prompt as well would confound it. A base model that cannot
    hold the format under the same instructions the fine-tune was trained on
    is a result, not a handicap to be corrected for.
    """
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


def _free_gpu(note=""):
    """Give the weights back to the driver.

    --base and --model load two 3B models in one process. Each is about
    2.2 GB in 4-bit, and a 4 GB card holds exactly one, so the second
    from_pretrained() fails with "Some modules are dispatched on the CPU or
    the disk" - which reads like the model is too big, when really the
    previous one is still resident.

    Dropping the Python reference is not enough on its own: the CUDA caching
    allocator keeps the block reserved, and unsloth's patching holds cycles
    that only a collection breaks. Hence gc first, then empty_cache.
    """
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
    """Run one model and return its raw output text per sentence.

    `adapter` is either a local adapter folder or the base-model id, and the
    system prompt travels with it so a base-model run cannot silently borrow
    the fine-tune's.

    Kept separate from alignment and cached to disk: a GPU pass costs
    minutes, and every change to the scoring logic would otherwise force
    another one.
    """
    FastLanguageModel, get_chat_template = _import_unsloth()
    if system_prompt is None:
        system_prompt = prompt_for(adapter)

    _free_gpu("before load")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter), max_seq_length=MAX_SEQ_LEN,
        dtype=None, load_in_4bit=True)
    tokenizer = get_chat_template(tokenizer, chat_template=template_for(adapter))
    FastLanguageModel.for_inference(model)

    # Llama 3.2 ships generation_config.max_length = 131072, and we pass
    # max_new_tokens instead. transformers 5.x warns about the pair on EVERY
    # call, which buries the progress output under 150 identical paragraphs.
    # Clearing the one we are not using silences it without changing behaviour
    # - max_new_tokens already took precedence.
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
    """Align generations to gold tokens; report the format-failure rate.

    THE ANSWER IS A DERIVATION, NOT A SENTENCE
    ------------------------------------------
    This used to do `text.split()` and align those words positionally. That was
    right for v1, whose whole answer was a bare lemma sentence with one word per
    token. The v2 answer is a table:

        <lema>
        itulah | IMBUHAN? | itu
        learn  | INGGERIS | learn
        </lema>
        <ayat>... </ayat>

    Splitting that on whitespace yields three words per token plus the tags, so
    the word count NEVER matches and every well-formed answer was reported as a
    format failure -- 100% of them -- while the scores came from LCS salvage
    over the tag soup. The model was fine; the metric was reading v2 output with
    a v1 ruler.

    So the block is parsed. A model that produced one derivation line per input
    token has succeeded, whatever its word count.
    """
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
    """The <ayat> sentence from each answer, for WER/CER.

    Measuring the raw generation against the reference sentence charged the
    reasoning models for every derivation line they were TRAINED to emit --
    which is why a fine-tuned model scored WER 2.65 against its own base
    model's 1.23. The sentence is the thing comparable to the reference; the
    derivation is scored separately, per token.
    """
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
# SEQUENCE METRICS
# ---------------------------------------------------------------------------

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
    """Corpus-level WER and CER of the produced lemma sentence.

    Corpus-level rather than the mean of per-sentence rates: a one-word
    sentence would otherwise carry the same weight as a thirty-word one.

    The hypothesis is the raw generation where there is one, so that words the
    alignment absorbed still count as insertions or deletions. For the rule
    baselines, which emit one lemma per token, it is those lemmas joined.
    """
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
    """Exact two-sided McNemar on paired binary outcomes.

    The right test here: both systems are scored on the SAME tokens, so the
    outcomes are paired and only the disagreements carry information.
    Returns (b-only-correct, a-only-correct, p).
    """
    from math import comb
    n01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    n10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    # Integer 2, not 2.0. base-llama disagrees with rules on ~1,500 of the
    # 1,786 tokens, and comb(1500, 750) has ~450 digits: multiplying that sum
    # by a float converts it to a float first and raises OverflowError. Kept
    # exact until the final int/int division, which Python rounds correctly at
    # arbitrary precision and underflows to 0.0 rather than raising.
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
    """The 300 dev sentences, with their silver annotation attached.

    dev_300.jsonl is ChatML and carries no per-token labels, so the ids are
    read from the split manifest and the annotated rows pulled from
    silver.jsonl. Scoring needs `lemmas`, `band` and `src`, not just text.
    """
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
    """Every token the two systems disagree on, with the gold answer.

    The aggregate gap between `rules` and `finetune` is 31 tokens out of
    1,786 - small enough to print in full, which is the only way to know
    whether it is a systematic weakness or a handful of accidents. Printing
    all of them, not a chosen few, is the point: a report that quotes its
    best three examples is not evidence.

    Uses the same predictions the scores above were computed from, so this
    cannot drift from the tables.
    """
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
        # The human-labelled VALIDATION split. This is where the choice
        # between models is made, and it exists precisely because neither of
        # the other two splits can make it: the dev split is silver, so the
        # rules grade their own homework there, and gold is the test set --
        # selecting on it and then reporting its score would mean the reported
        # number no longer estimates anything.
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

    # --reuse means "score what has already been generated", so it should pick
    # the base cache up too. Without this it silently produced a table with no
    # before-fine-tuning row while the predictions sat on disk unread.
    # The four-way comparison needs BOTH un-fine-tuned bases, not just Llama's.
    # Each is generated once and cached by name, so a later run picks it up from
    # disk with --reuse instead of paying for it again -- which matters when a
    # 4 GB card holds exactly one 3B model at a time and each pass is an hour.
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

    # Every adapter asked for, so both fine-tunes can share one table (and one
    # McNemar comparison) instead of being read off two separate runs.
    for ad in adapters:
        if with_model or (reuse and preds_path(ad).exists()):
            run_neural(label_for(ad), rows, ad, reuse, results, fmt_fail)

    # --models-only narrows every table below to the neural systems. The
    # non-neural rows are still COMPUTED (they cost nothing and the honest-
    # subset section needs `rules` to have a reference), they are just not
    # printed -- so a run that reports four models can still be compared
    # against a run that reports eight, from the same cached predictions.
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

    # ---- head to head, for choosing between the fine-tunes ----------------
    #
    # Every test above is against `rules`, which answers "is this system worth
    # anything". Model SELECTION asks a different question -- which of the two
    # fine-tunes is better -- and that comparison is not in the table above.
    # It needs its own test because the two are paired on the same tokens and
    # will usually be close: a raw accuracy gap between them is meaningless
    # until you know the tokens they disagree on are lopsided.
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
