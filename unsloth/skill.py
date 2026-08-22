#!/usr/bin/env python
"""
The lemmatizer's SKILL: what the model is told, what it must avoid, and the
shape of what it answers.

    python skill.py --card                    # print the system prompt
    python skill.py --rules                   # the rule vocabulary, with glosses
    python skill.py --demo                    # render real rows into targets
    python skill.py --demo --seed ../data/gold.jsonl -n 5

WHY THIS FILE EXISTS
--------------------
v1 of this model answered with a bare lemma sentence ("Kembalikan ayat
sahaja"). That is unauditable: when it returns "jalan" for "berjalan" you
cannot tell whether it stripped a prefix, hit a lexicon entry, or guessed. v2
makes the model show its derivation, one line per token.

The reasoning is NOT free text. Each line carries a tag from a closed
vocabulary (RULES below) naming the rule that produced the lemma. That matters
for three reasons:

  1. It is gradeable. A tag can be scored against the `src` field that
     data/gold.jsonl, data/silver.jsonl and data/val_seed.jsonl already carry.
     Free-form prose cannot.
  2. It is cheap. A 3B model asked to write explanations will write fluent
     wrong ones; asked to pick one of nine tags, it mostly picks right.
  3. It is already labelled. databuilder.lemma_of() returns `src` -- the rule
     that fired -- for every token it annotates. So the training targets are
     rendered from provenance the corpus already carries, not invented.

The prose explanation a human reads is generated at DISPLAY time, from the tag
(see explain()). The model never writes prose, so it can never write prose that
contradicts what it did.

VERSIONING
----------
SYSTEM_PROMPT must be byte-identical between training and inference; a stale
copy is what once made every answer nonsense (see chat.py). v1 is kept below
because lemmatizer_lora/ was trained on it and still needs it. Do not edit a
version in place -- add a new one and a new ADAPTER_DIR.

This module is stdlib-only on purpose: it must stay runnable without unsloth,
torch or a GPU, so the skill can be inspected and tested before any retrain.
"""

import argparse
import json
import os
import re

# ---------------------------------------------------------------------------
# VERSIONS
# ---------------------------------------------------------------------------

# v1: bare lemma sentence. The adapter in lemmatizer_lora/ was trained on this
# exact string. Kept so that adapter stays loadable; do not modify.
SYSTEM_PROMPT_V1 = (
    "Anda adalah pelemma bahasa Melayu. Tukarkan ayat rojak kepada ayat "
    "berbentuk lema. Kembalikan ayat sahaja."
)

SKILL_VERSION = "v2-reasoning"
ADAPTER_DIR = "lemmatizer_lora_v2"


# ---------------------------------------------------------------------------
# RULE VOCABULARY
# ---------------------------------------------------------------------------
#
# Nine tags, chosen to be the smallest set that still separates the error modes
# worth telling apart. They collapse databuilder's eleven `src` values: the
# `_guess` variants are not separate rules, they are the same rule applied with
# less evidence, which the "?" uncertainty marker carries instead.

RULES = {
    "KAMUS":    "kata dasar, sudah berbentuk lema",
    "RINGKAS":  "singkatan atau slanga, dikembangkan",
    "IMBUHAN":  "imbuhan dibuang untuk dapat kata dasar",
    # SETTLED: both halves now resolve reduplication to its base (budak2 ->
    # budak, lagu-lagu -> lagu) EXCEPT kata ganda semu, which is only ever a
    # word doubled ("rama-rama" is a butterfly; "rama" is not a small one).
    # Those keep the doubled form. The exceptions are listed in
    # malindo_morph/ganda_semu.txt, read by both databuilder.lemma_of and
    # Dictionary.morphological_analysis so the two cannot drift apart again.
    "GANDA":    "penggandaan diselesaikan kepada kata dasar, "
                "kecuali kata ganda semu yang kekal berganda",
    "PANJANG":  "huruf berulang dipendekkan",
    "PARTIKEL": "partikel wacana, sengaja tidak ditukar",
    "INGGERIS": "kata Inggeris, dikekalkan",
    "NOMBOR":   "nombor, masa, tarikh atau wang",
    "KEKAL":    "tidak dikenali, dikekalkan seadanya",
}

# `src` -> (rule tag, uncertain).
#
# There are TWO source vocabularies in this project and they do not overlap
# much. Both are mapped here, because a tag that silently degrades to KEKAL is
# worse than no tag at all -- it teaches the model to shrug.
#
#   A. unsloth/databuilder.py  -> unsloth/data/val_seed.jsonl
#   B. crawl/make_goldsheet.py -> data/gold.jsonl, data/silver.jsonl
#      (definitions quoted from make_goldsheet.py's module docstring)
SRC_TO_RULE = {
    # --- A: databuilder.lemma_of() ---
    "dict":             ("KAMUS", False),
    "slang":            ("RINGKAS", False),
    "particle":         ("PARTIKEL", False),
    "reduplication":    ("GANDA", False),
    "elongation":       ("PANJANG", False),
    "number":           ("NOMBOR", False),
    "money":            ("NOMBOR", False),
    "time":             ("NOMBOR", False),
    "date":             ("NOMBOR", False),
    "percent":          ("NOMBOR", False),
    "codeswitch":       ("INGGERIS", False),
    "affix":            ("IMBUHAN", True),
    "affix_guess":      ("IMBUHAN", True),
    "codeswitch_guess": ("INGGERIS", True),
    "ms_unlisted":      ("KEKAL", True),
    "unknown":          ("KEKAL", True),

    # --- B: make_goldsheet.guess() ---
    # `affix`, `dict`, `slang`, `particle` are shared with A above.
    "num":              ("NOMBOR", False),   # "a bare number, left alone"
    "redup":            ("GANDA", False),    # "budak2 -> budak"
    "elong":            ("PANJANG", False),  # "gilaaa -> gila"
    # "MALINDO gave several lemmas, picked 1" -- a real dictionary lemma, but
    # the choice between candidates is exactly what needs checking.
    "ambiguous":        ("KAMUS", True),
    # "MALINDO echoed an informal spelling back at us": lemma == token and the
    # token looks nonstandard. Tagging that KAMUS would teach the model that
    # informal spellings are roots, so it is KEKAL -- kept, not resolved.
    "dict_check":       ("KEKAL", True),
    "oov":              ("KEKAL", True),     # "nothing resolved it"
    # "human" is not a rule. make_goldsheet.py overwrites the original rule
    # when an annotator corrects a lemma, so the WHICH-rule information is
    # gone while the lemma itself became the most trustworthy in the project.
    # Handled by rule_for() below, which re-derives the rule from the surface
    # pair instead of dumping 132 hand-checked gold tokens into KEKAL.
}

# English gloss, for reports and for readers who do not read Malay.
RULES_EN = {
    "KAMUS":    "already a root; unchanged",
    "RINGKAS":  "abbreviation/slang, expanded",
    "IMBUHAN":  "affix stripped to reach the root",
    "GANDA":    "reduplication resolved to its base",
    "PANJANG":  "elongated letters collapsed",
    "PARTIKEL": "discourse particle, deliberately untouched",
    "INGGERIS": "English token, kept as-is",
    "NOMBOR":   "number, time, date or money",
    "KEKAL":    "unrecognised, left alone",
}


# ---------------------------------------------------------------------------
# THE SKILL
# ---------------------------------------------------------------------------
#
# Every "JANGAN" below is a failure this pipeline actually produces, not a
# generic caution:
#   - translating is the base model's default reading of "convert this Malay"
#   - answering the question is what OFF-TASK probes trigger (test_model.py)
#   - one-line-per-token is load-bearing: llm.align() gives up when the token
#     count drifts, and a misaligned row scores zero on every token
#   - particles: PARTICLES in databuilder is a deliberate do-not-normalise set
#   - NEVER_STRIP exists because "dengan" -> "gan" and "makan" -> "kan" are
#     what a naive affix stripper does
#   - splitting "budak2" into "budak budak" breaks alignment AND the label

SYSTEM_PROMPT = """\
Anda pelemma bahasa Melayu rojak. Untuk setiap token, beri lema dan sebabnya.

Format jawapan, tiada teks lain:
<lema>
token | PERATURAN | lema
</lema>
<ayat>ayat lema penuh</ayat>

PERATURAN:
KAMUS    kata dasar, kekal
RINGKAS  singkatan/slanga dikembangkan (dh->sudah, xleh->tidak boleh)
IMBUHAN  imbuhan dibuang (berjalan->jalan, memakan->makan)
GANDA    penggandaan jadi satu lema (budak2->budak)
PANJANG  huruf berulang dipendekkan (sedappp->sedap)
PARTIKEL partikel wacana, kekal (la, weh, kan, je)
INGGERIS kata Inggeris, kekal
NOMBOR   nombor, masa, tarikh, wang
KEKAL    tidak dikenali, kekal
Tulis "?" selepas peraturan jika tidak pasti, contoh: INGGERIS?

JANGAN:
- jangan terjemah ayat ke bahasa Inggeris
- jangan jawab soalan atau beri pendapat; lemakan sahaja
- satu baris bagi setiap token, ikut susunan asal, jangan tambah atau buang
- jangan tukar partikel atau nama khas
- jangan buang imbuhan daripada kata dasar (dengan, makan, kerana, sebab)
- jangan pecahkan penggandaan kepada dua perkataan
- jangan reka lema untuk kata yang tidak dikenali
- jangan tulis apa-apa di luar <lema> dan <ayat>"""


# ---------------------------------------------------------------------------
# RENDER: annotations -> training target
# ---------------------------------------------------------------------------

DIGIT_REDUP_RE = re.compile(r"^(.+?)2$")
ELONG_RUN_RE = re.compile(r"(.)\1{2,}")


def infer_rule(token, lemma):
    """Re-derive the rule from a token/lemma pair, for labels that lost theirs.

    Only used for `src == "human"`, where an annotator's correction overwrote
    the rule that produced the lemma. Surface evidence recovers the mechanical
    cases exactly; anything left is a change a person made for a reason the
    strings do not show, so it is reported as RINGKAS with the uncertainty
    marker set -- the lemma is trustworthy, the rule label is a guess.
    """
    t, l = token.lower(), lemma.lower()
    if t.isdigit():
        return "NOMBOR", False
    if DIGIT_REDUP_RE.match(t) or (("-" in t) and t.partition("-")[0] == t.partition("-")[2]):
        return "GANDA", False
    if ELONG_RUN_RE.search(t):
        return "PANJANG", False
    if t == l:
        return "KEKAL", False
    return "RINGKAS", True


# Prose -> tag. Some corpora explain a lemma in words instead of labelling it
# ("suffix -an (noun-forming) removed"). The rule vocabulary recognises itself
# in that prose, so those corpora can be folded in without a second format.
# Patterns are ordered: the first match wins, and the order encodes precedence.
# "-kah" must be read as a suffix before the word "particle" in the same
# sentence pulls it towards PARTIKEL.
_REASON_PATTERNS = [
    ("KEKAL",    r"proper noun|nama khas|left in its dictionary form"),
    ("GANDA",    r"reduplicat|penggandaan|plural(?:is|ity)?\b|repeated word|doubling"),
    ("PANJANG",  r"elongat|lengthen|repeated letter|drawn[- ]out|extra letter|stretch"),
    ("NOMBOR",   r"\bnumber\b|numeral|\bdigit|\btime\b|\bdate\b|money|currency"),
    ("IMBUHAN",  r"prefix|suffix|circumfix|confix|infix|clitic|applicative|causative"
                 r"|nominalis|verbalis|\bmeN|\bpeN|\bber-|\bter-|\bdi-|\bke-|\bse-"
                 r"|\bmem|\bmeng|\bmeny|\bmen-|\bpem|\bpeng|\bpeny"
                 r"|-an\b|-kan\b|-nya\b|-kah\b|-lah\b|-tah\b|-i\b"),
    ("INGGERIS", r"english|loanword|borrow|code[- ]?switch"),
    ("RINGKAS",  r"slang|shorthand|sms|informal|colloquial|spoken|clipping|clipped"
                 r"|everyday|standard form|non[- ]?standard|spelling|abbreviat"
                 r"|contract|texting|dialect|short for|standardised|standardized"),
    ("PARTIKEL", r"particle"),
]
_REASON_RE = [(tag, re.compile(pat, re.I)) for tag, pat in _REASON_PATTERNS]


def classify_reason(prose):
    """Map a free-text explanation to a rule tag, or None if nothing matches.

    Returns (tag, uncertain). `uncertain` is always False: the prose was
    written by whoever built that corpus and names the affix explicitly, which
    is stronger evidence than databuilder's rule guesses, not weaker.
    """
    for tag, rx in _REASON_RE:
        if rx.search(prose):
            return tag, False
    return None


def rule_for(src, token=None, lemma=None):
    """Map a `src` to (tag, uncertain).

    `token`/`lemma` are only consulted for labels carrying no rule of their
    own. An unrecognised src degrades to KEKAL? rather than raising, so a new
    category in a future dataset shows up as visibly unhandled instead of
    crashing a training run.
    """
    if src == "human" and token is not None and lemma is not None:
        return infer_rule(token, lemma)
    return SRC_TO_RULE.get(src, ("KEKAL", True))


def render_target(row):
    """Build the assistant turn for one annotated row.

    The <ayat> line is rebuilt from the per-token lemmas rather than copied
    from row["lemma_text"], so the derivation and the sentence can never
    disagree in training data. If they could, the model would learn that they
    are allowed to.
    """
    lines = []
    for e in row["lemmas"]:
        # A corpus that already knows its rule carries it as "rule"/"unsure";
        # everything else is derived from the `src` provenance.
        if e.get("rule"):
            tag, uncertain = e["rule"], bool(e.get("unsure"))
        else:
            tag, uncertain = rule_for(e.get("src"), e["t"], e["l"])
        lines.append("%s | %s%s | %s" % (e["t"], tag, "?" if uncertain else "", e["l"]))
    sentence = " ".join(e["l"] for e in row["lemmas"])
    return "<lema>\n%s\n</lema>\n<ayat>%s</ayat>" % ("\n".join(lines), sentence)


def to_chatml(row, system_prompt=SYSTEM_PROMPT):
    return {"messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": row["text"]},
        {"role": "assistant", "content": render_target(row)},
    ]}


# ---------------------------------------------------------------------------
# PARSE: model output -> structured result
# ---------------------------------------------------------------------------

LEMA_BLOCK_RE = re.compile(r"<lema>(.*?)</lema>", re.S | re.I)
AYAT_BLOCK_RE = re.compile(r"<ayat>(.*?)</ayat>", re.S | re.I)
# "token | TAG? | lemma". The lemma may be multi-word ("tidak boleh").
LINE_RE = re.compile(r"^\s*(\S+)\s*\|\s*([A-Z]+)(\??)\s*\|\s*(.+?)\s*$")


def parse_response(text):
    """Read a model answer back into structured form.

    Returns a dict with:
        steps        [{token, rule, uncertain, lemma}, ...]
        lemma_text   the <ayat> sentence, or the derivation joined if absent
        ok           the answer had both blocks and at least one parsed step
        consistent   <ayat> agrees with the lemmas in <lema>
        bad_rules    tags outside RULES (model invented a rule)

    Deliberately lenient about everything except the parts that carry meaning:
    a fine-tune that drops the closing tag on long outputs is still telling you
    what it derived, and throwing that away would hide a mostly-good answer.
    """
    block = LEMA_BLOCK_RE.search(text)
    # Tolerate a missing </lema> when the model ran out of tokens mid-answer.
    if block:
        body = block.group(1)
    elif "<lema>" in text.lower():
        body = text[text.lower().index("<lema>") + 6:].split("<ayat>")[0]
    else:
        body = text

    steps, bad_rules = [], []
    for line in body.splitlines():
        m = LINE_RE.match(line)
        if not m:
            continue
        token, rule, mark, lemma = m.groups()
        if rule not in RULES:
            bad_rules.append(rule)
        steps.append({
            "token": token,
            "rule": rule,
            "uncertain": mark == "?",
            "lemma": lemma.strip(),
        })

    derived = " ".join(s["lemma"] for s in steps)
    ayat = AYAT_BLOCK_RE.search(text)
    lemma_text = ayat.group(1).strip() if ayat else derived

    return {
        "steps": steps,
        "lemma_text": lemma_text,
        "ok": bool(steps) and ayat is not None,
        "consistent": bool(steps) and _norm(lemma_text) == _norm(derived),
        "bad_rules": bad_rules,
    }


def _norm(s):
    return re.sub(r"\s+", " ", s.strip().lower())


# ---------------------------------------------------------------------------
# EXPLAIN: tag -> prose, at display time
# ---------------------------------------------------------------------------

def explain(rule, uncertain=False, lang="ms"):
    gloss = (RULES if lang == "ms" else RULES_EN).get(rule, "peraturan tidak dikenali")
    if uncertain:
        gloss += " (tidak pasti)" if lang == "ms" else " (low confidence)"
    return gloss


def format_result(parsed, lang="ms"):
    """Render a parsed answer as an aligned table plus the final sentence."""
    if not parsed["steps"]:
        return "(tiada analisis boleh dibaca)"

    w_tok = max(len(s["token"]) for s in parsed["steps"])
    w_lem = max(len(s["lemma"]) for s in parsed["steps"])
    out = []
    for s in parsed["steps"]:
        tag = s["rule"] + ("?" if s["uncertain"] else "")
        out.append("  %-*s  ->  %-*s  %-9s %s" % (
            w_tok, s["token"], w_lem, s["lemma"], tag,
            explain(s["rule"], s["uncertain"], lang)))

    out.append("")
    out.append("  " + parsed["lemma_text"])
    if not parsed["consistent"]:
        out.append("  [amaran] <ayat> tidak sepadan dengan analisis token")
    if parsed["bad_rules"]:
        out.append("  [amaran] peraturan tidak dikenali: %s" % ", ".join(parsed["bad_rules"]))
    return "\n".join(out)


# ---------------------------------------------------------------------------

def _load_seed(path=None):
    """Any file whose rows carry `text` + `lemmas[{t,l,src}]` works here, which
    includes the hand-labelled ../data/gold.jsonl, not just the mined seeds."""
    candidates = [path] if path else [
        "data/val_seed.jsonl", "seed500.jsonl", "seed.jsonl",
        "../data/gold.jsonl", "../data/silver.jsonl",
    ]
    for p in candidates:
        if p and os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return p, [json.loads(l) for l in f if l.strip()]
    raise SystemExit("no annotated file found; pass --seed PATH")


def _selftest(rows):
    """Every rendered target must parse back to exactly what produced it."""
    bad = []
    for r in rows:
        p = parse_response(render_target(r))
        want = [(e["rule"] if e.get("rule") else rule_for(e.get("src"), e["t"], e["l"])[0])
                for e in r["lemmas"]]
        if not (p["ok"] and p["consistent"]
                and [s["lemma"] for s in p["steps"]] == [e["l"] for e in r["lemmas"]]
                and [s["rule"] for s in p["steps"]] == want):
            bad.append(r.get("id", "?"))
    return bad


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--card", action="store_true", help="print the system prompt")
    p.add_argument("--rules", action="store_true", help="print the rule vocabulary")
    p.add_argument("--demo", action="store_true", help="render annotated rows as targets")
    p.add_argument("--seed", help="path to an annotated .jsonl")
    p.add_argument("-n", type=int, default=3)
    a, _ = p.parse_known_args()

    if a.rules:
        print("%-9s %-44s %s" % ("TAG", "MALAY", "ENGLISH"))
        print("-" * 100)
        for tag, ms in RULES.items():
            print("%-9s %-44s %s" % (tag, ms, RULES_EN[tag]))
        print("\nsrc -> tag")
        for src, (tag, unc) in SRC_TO_RULE.items():
            print("  %-18s %s%s" % (src, tag, "?" if unc else ""))
    elif a.demo:
        path, rows = _load_seed(a.seed)
        print("source: %s (%d rows)\n" % (path, len(rows)))
        for row in rows[:a.n]:
            print("=" * 72)
            print("INPUT :", row["text"])
            print("-" * 72)
            print(render_target(row))
            print()
        bad = _selftest(rows)
        print("=" * 72)
        print("self-test: %d/%d rows render and parse back exactly%s"
              % (len(rows) - len(bad), len(rows),
                 "" if not bad else "  FAILED: %s" % bad[:5]))
    else:
        print(SYSTEM_PROMPT)


if __name__ == "__main__":
    main()
