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
import csv
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

# Which skill version each adapter on disk was trained with.
#
# chat.py used to take its system prompt from llm.py, which picks v1 or v2 off
# the USE_REASONING environment variable. That variable describes a TRAINING
# run, not the adapter you are loading, so with it unset (the default, "1")
# `chat.py --adapter lemmatizer_lora_yt` handed the v2 reasoning prompt to a v1
# adapter -- the prompt/adapter mismatch chat.py's own docstring warns about.
# The adapter decides now, and this table is where it decides from.
ADAPTERS = {
    "lemmatizer_lora":           "v1",   # first seed500 run
    "lemmatizer_lora_yt":        "v1",   # YouTube crawl run
    "lemmatizer_lora_yt_prev":   "v1",
    "lemmatizer_lora_yt_backup": "v1",
    "lemmatizer_lora_v2":        "v2",   # == ADAPTER_DIR above
}

# Written into an adapter folder at save time so it describes itself, and a
# copied or renamed folder still knows which prompt it needs.
SKILL_FILE = "skill.json"


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
# WHICH SKILL DOES THIS ADAPTER SPEAK?
# ---------------------------------------------------------------------------

PROMPTS = {"v1": SYSTEM_PROMPT_V1, "v2": SYSTEM_PROMPT}


def version_for(adapter_dir):
    """Which skill version an adapter expects: "v1" or "v2".

    Three sources, most trustworthy first:
      1. <adapter>/skill.json, stamped by llm.py when the adapter was saved.
      2. ADAPTERS, for the four adapters that predate that stamp.
      3. The folder name, with a warning. Guessing wrong here produces the
         failure that is hardest to notice: the model still answers fluently,
         and the answer is nonsense. Better to say out loud that we guessed.
    """
    adapter_dir = str(adapter_dir)
    name = os.path.basename(os.path.normpath(adapter_dir))

    meta = os.path.join(adapter_dir, SKILL_FILE)
    if os.path.exists(meta):
        try:
            with open(meta, encoding="utf-8") as fh:
                v = json.load(fh).get("skill_version")
            if v in PROMPTS:
                return v
        except (OSError, ValueError):
            pass            # unreadable stamp is no worse than no stamp

    if name in ADAPTERS:
        return ADAPTERS[name]

    guess = "v2" if "v2" in name else "v1"
    print("[skill] %r is not in skill.ADAPTERS -- guessing %s from the name. "
          "Add it there to make this certain." % (name, guess))
    return guess


def prompt_for(adapter_dir):
    """(system prompt, version) for an adapter.

    The only supported way to get a system prompt at inference time. chat.py
    once carried its own retyped copy, it went stale, and every answer turned
    to nonsense; asking here means that cannot happen again.
    """
    v = version_for(adapter_dir)
    return PROMPTS[v], v


def write_version(adapter_dir, version):
    """Stamp a freshly saved adapter with the skill version it was trained on."""
    if version not in PROMPTS:
        raise ValueError("unknown skill version %r" % (version,))
    path = os.path.join(adapter_dir, SKILL_FILE)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"skill_version": version, "skill_name": SKILL_VERSION}, fh,
                  indent=2)
    return path


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
        # These tags are the MODEL's claim about what it did. derive_steps()
        # returns the same shape with derived=True, where the tags are ours.
        "derived": False,
    }


def _norm(s):
    return re.sub(r"\s+", " ", s.strip().lower())


# ---------------------------------------------------------------------------
# DERIVE: recover a derivation from a v1 answer
# ---------------------------------------------------------------------------
#
# v1 adapters (lemmatizer_lora, lemmatizer_lora_yt) answer with a bare lemma
# sentence. They cannot be prompted into explaining themselves -- the output
# format is fine-tuned in, and asking for reasoning only breaks the format
# (see for yourself: chat.py --skill v2). Retraining is the real fix. Until
# then, rather than show nothing, we re-derive the rule for each token from the
# (token, lemma) pair the model actually produced.
#
# This is NOT the model's reasoning and must never be shown as if it were. The
# claim it makes is weaker and precise: "this is the rule that accounts for the
# change the model made". That is enough to audit an answer by hand, and it is
# checkable -- every tag traces to a test over the two surface strings and the
# same lexicons databuilder annotated the corpus with. format_result() labels
# derived tags, so a reader can always tell the two apart.

_DB = None


def _databuilder():
    """databuilder, or None if it cannot be imported.

    Lazy and optional on purpose: skill.py must keep printing its card, its
    rules and its training targets on a machine that has this file and nothing
    else. Only the derivation below needs the lexicons.
    """
    global _DB
    if _DB is None:
        try:
            import databuilder
            # EN_VOCAB starts empty and is filled from the corpus, which is
            # how databuilder decided "codeswitch" when it labelled the data.
            # Without this the INGGERIS tag can only fire on the ~50 words in
            # EN_HINT, and every other English token gets shrugged at.
            try:
                databuilder.induce_from_uploads(quiet=True)
            except OSError:
                pass            # no sentidata/ here; EN_HINT alone, then
            _DB = databuilder
        except Exception:
            _DB = False
    return _DB or None


def align_lemmas(tokens, words):
    """Split a lemma sentence back into one lemma per input token.

    A lemma can be several words ("xleh" -> "tidak boleh"), so the two lists do
    not line up 1:1. The slang table knows how many words each token is
    expected to become; if that total matches, use it. Plain 1:1 is the
    fallback. Anything else is unsafe -- llm.align() takes the same position --
    and returns None, so the caller can report the mismatch instead of printing
    a table in which every row after the drift is wrong.
    """
    db = _databuilder()
    slang = getattr(db, "SLANG", {}) if db is not None else {}

    widths = [len(slang[t.lower()].split()) if t.lower() in slang else 1
              for t in tokens]
    if sum(widths) != len(words):
        if len(tokens) != len(words):
            return None
        widths = [1] * len(tokens)

    out, i = [], 0
    for w in widths:
        out.append(" ".join(words[i:i + w]))
        i += w
    return out


# MALINDO Morph, read for two things: which strings are roots, and which
# surface form comes from which root. databuilder.ROOTS is a 200-word starter
# list whose own comment says "Replace with MALINDO Morph output for real
# work", and a real root it does not carry gets tagged KEKAL? -- which reads as
# "we did not recognise this" when the truth is "we did not look it up".
#
# Read with stdlib csv rather than through malindo_morph.Dictionary: that class
# needs pandas, imports Preprocess.tokenizer, and resolves its path against the
# caller's cwd (so it breaks when run from unsloth/). None of that is allowed
# here. One pass costs ~0.6s and is taken once per process, lazily, so the
# commands that do not derive anything never pay it.
MALINDO_TSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           os.pardir, "malindo_morph", "malindo_dic_2024.tsv")

_MALINDO = None


# MALINDO's `source` column says where a row came from, and the distinction
# decides what the row may be used for. "Leipzig" is a scraped web corpus, so
# it carries English: "the", "of", "and", "share" and "time" are all in there
# as roots. Tagging those KAMUS would assert that an English function word is a
# Malay dictionary entry -- a positively wrong claim, not a shrug. Only the
# lexicon proper is allowed to establish roothood.
#
# The surface -> root mapping keeps every row, Leipzig included: corpus
# evidence that "berjalan" derives from "jalan" is still evidence, and being
# wrong there would need the pair to be wrong, not merely non-dictionary.
DICT_SOURCES = {"Kamus"}


def malindo():
    """(surface -> {roots}, {dictionary roots}), or (None, None) if TSV absent.

    Missing is not an error: the derivation below falls back to databuilder's
    lexicons and marks more tags uncertain, which is the correct behaviour for
    a checkout that did not pull the 16MB dictionary.
    """
    global _MALINDO
    if _MALINDO is None:
        surfaces, roots = {}, set()
        try:
            with open(MALINDO_TSV, encoding="utf-8", newline="") as fh:
                for row in csv.reader(fh, delimiter="\t"):
                    if len(row) > 7 and row[1] and row[2]:
                        root, surface = row[1].lower(), row[2].lower()
                        surfaces.setdefault(surface, set()).add(root)
                        if row[7] in DICT_SOURCES:
                            roots.add(root)
            _MALINDO = (surfaces, roots)
        except OSError:
            _MALINDO = (None, None)
    return _MALINDO


def derive_rule(token, lemma):
    """(tag, uncertain) for how `token` could have become `lemma`.

    Ordered strongest evidence first, because several tests fire on one token:
    "budak-budak" is a reduplication whose halves are also a known root, and
    "sedappp" has to collapse before any lexicon can recognise it. First match
    wins, so this order IS the precedence.

    `uncertain` is set when the tag is the best available account rather than a
    demonstrated one. It renders as the same "?" a v2 model writes, so both
    kinds of doubt read the same way to whoever is checking the output.
    """
    t = token.lower()
    l = lemma.lower().strip()
    db = _databuilder()

    # --- numbers, money, time, date, percent -------------------------------
    if db is not None:
        for rx in (db.MONEY_RE, db.TIME_RE, db.DATE_RE, db.PCT_RE, db.NUM_RE):
            if rx.match(t):
                return "NOMBOR", False
    elif t.replace(".", "").replace(",", "").isdigit():
        return "NOMBOR", False

    # --- reduplication -----------------------------------------------------
    # preprocess.py rewrites "budak2" as "budak-budak" before the model sees
    # it, so the doubled form is what is on the page. Both spellings are
    # checked anyway, for --raw input which skipped that stage.
    head, sep, tail = t.partition("-")
    if (sep and head == tail) or DIGIT_REDUP_RE.match(t):
        return "GANDA", False

    # --- elongation --------------------------------------------------------
    # Only when collapsing the run actually produces the lemma. An elongated
    # token the model did something ELSE to falls through to the tests below
    # and picks up its "?" at the bottom.
    if db is not None:
        collapsed = db.normalise_repeat(t)
        if collapsed is not None and collapsed == l:
            return "PANJANG", False
    elif ELONG_RUN_RE.search(t) and t != l:
        return "PANJANG", False

    # --- particles ---------------------------------------------------------
    # PARTICLES is a deliberate do-not-normalise set, so the tag is only
    # correct if the model did in fact leave the token alone.
    if db is not None and t in db.PARTICLES and t == l:
        return "PARTIKEL", False

    # --- slang / abbreviation ----------------------------------------------
    if db is not None and t in db.SLANG:
        # Known token, but the model expanded it differently from the corpus.
        # Still RINGKAS -- that is the mechanism -- flagged so it gets read.
        return "RINGKAS", db.SLANG[t] != l

    # --- affixes -----------------------------------------------------------
    surfaces, roots = malindo()

    # MALINDO derives this exact surface form from this exact root, so the
    # change the model made IS an affix operation. That is evidence, not a
    # guess, and it is the one tag in this function backed by a real
    # morphological dictionary rather than by string surgery.
    if surfaces is not None and t != l and l in surfaces.get(t, ()):
        return "IMBUHAN", False

    # strip_affix only fires when the root lands in databuilder.ROOTS, which is
    # why it is trusted. strip_affix_loose guesses, and its guess is accepted
    # only when it matches what the model said -- two independent routes to the
    # same string is better evidence than either alone.
    if db is not None and t != l:
        root, _ = db.strip_affix(t)
        if root == l:
            return "IMBUHAN", False
        loose, ok = db.strip_affix_loose(t)
        if ok and loose == l:
            return "IMBUHAN", False

    # --- unchanged ---------------------------------------------------------
    if t == l:
        # English is tested BEFORE roothood, because the two overlap and the
        # overlap is not symmetric: plenty of English words appear somewhere in
        # a Malay corpus, while EN_VOCAB only holds tokens the corpus saw
        # overwhelmingly in English lines. When both fire, English is the
        # better-evidenced of the two.
        if db is not None and (t in db.EN_HINT or t in db.EN_VOCAB):
            return "INGGERIS", False
        if roots is not None and t in roots:
            return "KAMUS", False
        if db is not None and t in db.ROOTS:
            return "KAMUS", False
        # An English word the hint list does not carry is indistinguishable
        # from a Malay word we have no root for: both were kept as they are,
        # which is what KEKAL says. Claiming INGGERIS here would be a guess
        # about the language, not a statement about the lemma.
        return "KEKAL", True

    # --- changed, and nothing above accounts for it ------------------------
    if ELONG_RUN_RE.search(t):
        return "PANJANG", True
    return "RINGKAS", True


def derive_steps(tokens, lemma_text):
    """A parse_response()-shaped result for a v1 answer, so both render alike.

    When the sentence will not align to the tokens, `steps` comes back empty
    and `unaligned` carries the two counts. The sentence is still returned: a
    lemma sentence we cannot break down is worth more than nothing, and the
    counts say exactly why the table is missing.
    """
    words = lemma_text.split()
    aligned = align_lemmas(tokens, words)

    if aligned is None:
        return {"steps": [], "lemma_text": lemma_text.strip(), "ok": False,
                "consistent": True, "bad_rules": [], "derived": True,
                "unaligned": (len(tokens), len(words))}

    steps = []
    for tok, lem in zip(tokens, aligned):
        rule, uncertain = derive_rule(tok, lem)
        steps.append({"token": tok, "rule": rule,
                      "uncertain": uncertain, "lemma": lem})

    return {"steps": steps, "lemma_text": lemma_text.strip(),
            "ok": bool(steps), "consistent": True, "bad_rules": [],
            "derived": True}


# ---------------------------------------------------------------------------
# EXPLAIN: tag -> prose, at display time
# ---------------------------------------------------------------------------

def explain(rule, uncertain=False, lang="ms"):
    gloss = (RULES if lang == "ms" else RULES_EN).get(rule, "peraturan tidak dikenali")
    if uncertain:
        gloss += " (tidak pasti)" if lang == "ms" else " (low confidence)"
    return gloss


# Every message the display can emit, in both languages, so adding one cannot
# quietly leave the other half untranslated.
NOTES = {
    "derived": (
        "peraturan diterbitkan daripada pasangan token->lema selepas jawapan; "
        "model v1 tidak menyatakan sebabnya sendiri",
        "rules derived from the token->lemma pairs after the fact; a v1 model "
        "does not state its own reasoning"),
    "unaligned": (
        "%d token masuk, %d perkataan keluar - tidak boleh dijajarkan, "
        "jadi tiada jadual peraturan",
        "%d tokens in, %d words out - cannot be aligned, so there is no "
        "rule table"),
    "unreadable": (
        "(tiada analisis boleh dibaca)",
        "(no readable analysis)"),
    "mismatch": (
        "<ayat> tidak sepadan dengan analisis token",
        "<ayat> disagrees with the per-token analysis"),
    "bad_rules": (
        "peraturan tidak dikenali: %s",
        "unknown rules: %s"),
}


def note(key, lang="ms"):
    return NOTES[key][0 if lang == "ms" else 1]


def format_result(parsed, lang="ms"):
    """Render a parsed answer as an aligned table plus the final sentence.

    Takes either shape: parse_response() output, where the tags are the
    model's own claim, or derive_steps() output, where they were re-derived
    here. The "[nota]" line below is what tells the reader which one they are
    looking at, and it is the whole reason derive_steps() sets a flag rather
    than passing itself off as a parse.
    """
    if not parsed["steps"]:
        if parsed.get("unaligned"):
            return "  %s\n\n  %s" % (note("unaligned", lang) % parsed["unaligned"],
                                     parsed["lemma_text"])
        return note("unreadable", lang)

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
        out.append("  [amaran] " + note("mismatch", lang))
    if parsed["bad_rules"]:
        out.append("  [amaran] " + note("bad_rules", lang) % ", ".join(parsed["bad_rules"]))
    if parsed.get("derived"):
        out.append("  [nota] " + note("derived", lang))
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


def _score(rows):
    """How often does a derived tag match the tag the corpus asserts?

    The model is taken out of the loop on purpose: each row is fed its own gold
    lemmas as if a v1 adapter had answered perfectly, so what is measured is
    the derivation alone, not the fine-tune. Anything below 100% here is a
    ceiling on what the derived table can tell you no matter how good the model
    gets.

    Read the per-tag recall, not the total. The six rules that explain a CHANGE
    -- KAMUS, RINGKAS, IMBUHAN, GANDA, PARTIKEL, NOMBOR -- score 95-100%,
    because each is decided by a test over the two strings plus a lexicon.
    INGGERIS and KEKAL are both "the model left this alone", and telling an
    unrecognised Malay word from an English one is language identification, a
    different problem from explaining a lemma. Most of that error is
    INGGERIS -> KEKAL, which is the safe direction: "unrecognised, left alone"
    understates rather than misstates.
    """
    import collections
    gold_n, hit_n = collections.Counter(), collections.Counter()
    conf = collections.Counter()
    unaligned = 0

    for r in rows:
        tokens = [e["t"] for e in r["lemmas"]]
        d = derive_steps(tokens, " ".join(e["l"] for e in r["lemmas"]))
        if not d["steps"]:
            unaligned += 1
            continue
        for step, e in zip(d["steps"], r["lemmas"]):
            gold = e["rule"] if e.get("rule") else rule_for(e.get("src"), e["t"], e["l"])[0]
            gold_n[gold] += 1
            if step["rule"] == gold:
                hit_n[gold] += 1
            else:
                conf[(gold, step["rule"])] += 1

    tot, agree = sum(gold_n.values()), sum(hit_n.values())
    out = ["rows: %d (unaligned: %d)" % (len(rows), unaligned),
           "derived tag matches the corpus label: %d/%d = %.1f%%"
           % (agree, tot, 100.0 * agree / tot if tot else 0.0), "",
           "%-9s %6s %6s %7s" % ("TAG", "gold", "right", "recall")]
    for tag in RULES:
        if gold_n[tag]:
            out.append("%-9s %6d %6d %6.0f%%"
                       % (tag, gold_n[tag], hit_n[tag],
                          100.0 * hit_n[tag] / gold_n[tag]))
    if conf:
        out += ["", "disagreements (corpus -> derived):"]
        out += ["  %-9s -> %-9s %4d" % (g, d, n) for (g, d), n in conf.most_common(6)]
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--card", action="store_true",
                   help="print the system prompt (add --adapter to pick which)")
    p.add_argument("--rules", action="store_true", help="print the rule vocabulary")
    p.add_argument("--demo", action="store_true", help="render annotated rows as targets")
    p.add_argument("--seed", help="path to an annotated .jsonl")
    p.add_argument("-n", type=int, default=3)
    p.add_argument("--adapter", help="which prompt this adapter gets")
    p.add_argument("--derive", nargs=2, metavar=("INPUT", "OUTPUT"),
                   help="derive the rule table for a v1 answer you paste in")
    p.add_argument("--score", action="store_true",
                   help="score derived tags against an annotated corpus")
    p.add_argument("--lang", default="ms", choices=("ms", "en"))
    a, _ = p.parse_known_args()

    if a.score:
        path, rows = _load_seed(a.seed)
        print("source: %s\n" % path)
        print(_score(rows))
    elif a.derive:
        # No GPU, no adapter, no model load: paste in what chat.py printed and
        # get the same table back. This is how the derivation is tested, and
        # how you demonstrate the skill on a machine that cannot run the model.
        src, out = a.derive
        print(format_result(derive_steps(src.split(), out), a.lang))
    elif a.adapter:
        prompt, v = prompt_for(a.adapter)
        print("adapter: %s\nskill:   %s\n%s\n%s" % (a.adapter, v, "-" * 72, prompt))
    elif a.rules:
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
