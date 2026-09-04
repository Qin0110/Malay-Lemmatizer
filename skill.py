#!/usr/bin/env python
"""
The lemmatiser's SKILL: what the model is told, what it must avoid, and the
shape of the answer it has to produce.

    python skill.py --card                 # print the system prompt
    python skill.py --rules                # the rule vocabulary, with glosses
    python skill.py --parse response.txt   # parse a model answer
    python skill.py --demo                 # render corpus rows into targets
    python skill.py --demo --seed data/gold.jsonl -n 3

WHY THIS FILE EXISTS
--------------------
A lemmatiser that answers with a bare sentence is unauditable. When it returns
"jalan" for "berjalan" there is no way to tell whether it stripped a prefix,
hit a dictionary entry, or guessed -- and those three failures need three
different fixes. This skill makes the model show its derivation, one line per
token.

The reasoning is NOT free text. Each line carries a tag from the closed
vocabulary in RULES, naming the rule that produced the lemma. That matters for
three reasons:

  1. It is gradeable. A tag can be scored against the `src` field the corpus
     already carries on every token. Free-form prose cannot.
  2. It is cheap. A 3B model asked to write explanations writes fluent wrong
     ones; asked to pick one of nine tags, it mostly picks right.
  3. It is already labelled. The corpus records which rule produced each
     silver lemma, so training targets are RENDERED from provenance the data
     already has, rather than invented after the fact.

The prose a human reads is generated at DISPLAY time from the tag (see
explain()). The model never writes prose, so it can never write prose that
contradicts what it actually did.

THE PROMPT IS PART OF THE MODEL
-------------------------------
SYSTEM_PROMPT must be byte-identical between training and inference. A model
fine-tuned on one prompt and served another still answers fluently -- it just
answers in the wrong format, and the accuracy drops for a reason that has
nothing to do with the weights. That failure is silent, which is why every
component imports the prompt from here instead of keeping its own copy. Do not
edit a version in place: add a new one, and stamp the adapters that use it.

Stdlib-only on purpose: this has to stay inspectable and testable without
torch, unsloth or a GPU, so the skill can be reviewed before any training run.
"""

import argparse
import json
import os
import re
import sys

SKILL_VERSION = "v1-reasoning"

# Written into an adapter folder at save time so the adapter describes itself,
# and a copied or renamed folder still knows which prompt it needs.
SKILL_FILE = "skill.json"


# ---------------------------------------------------------------------------
# RULE VOCABULARY
# ---------------------------------------------------------------------------
#
# Nine tags: the smallest set that still separates the error modes worth
# telling apart. Each one names a distinct thing that can go wrong, so a
# confusion matrix over these tags points at a specific fix.

RULES = {
    "KAMUS":    "kata dasar, sudah berbentuk lema",
    "RINGKAS":  "singkatan atau slanga, dikembangkan",
    "IMBUHAN":  "imbuhan dibuang untuk dapat kata dasar",
    "GANDA":    "penggandaan diselesaikan kepada kata dasar",
    "PANJANG":  "huruf berulang dipendekkan",
    "PARTIKEL": "partikel wacana, sengaja tidak ditukar",
    "INGGERIS": "kata Inggeris, dikekalkan",
    "NOMBOR":   "nombor, masa, tarikh atau wang",
    "KEKAL":    "tidak dikenali, dikekalkan seadanya",
}

# English gloss, for the report and for readers who do not read Malay.
RULES_EN = {
    "KAMUS":    "already a root; unchanged",
    "RINGKAS":  "abbreviation or slang, expanded",
    "IMBUHAN":  "affix stripped to reach the root",
    "GANDA":    "reduplication resolved to its base",
    "PANJANG":  "elongated letters collapsed",
    "PARTIKEL": "discourse particle, deliberately untouched",
    "INGGERIS": "English token, kept as-is",
    "NOMBOR":   "number, time, date or money",
    "KEKAL":    "unrecognised, left alone",
}

# Corpus `src` value -> (rule tag, uncertain).
#
# The second element is the uncertainty marker. It is not a separate rule: it
# is the SAME rule applied with less evidence, which is exactly what the "?"
# suffix means in the model's answer. Mapping an under-evidenced label to a
# confident tag would teach the model to be confident about guesses.
#
# Every `src` the corpus can emit is listed. A tag that silently degrades to
# KEKAL is worse than no tag at all -- it teaches the model to shrug.
SRC_TO_RULE = {
    "dict":       ("KAMUS",    False),   # MALINDO resolved it cleanly
    # A derived noun that IS its own dictionary headword: pengangkutan,
    # kecantikan, bacaan. Malay affixation is mostly derivational, so these
    # are separate lexemes, not forms of their root -- reducing them to
    # `angkut`/`cantik`/`baca` would be stemming, not lemmatisation. KAMUS
    # rather than IMBUHAN precisely because no affix should come off.
    "deriv":      ("KAMUS",    False),
    "slang":      ("RINGKAS",  False),   # from the SLANG lexicon
    "particle":   ("PARTIKEL", False),   # kept as-is on purpose
    "num":        ("NOMBOR",   False),   # a bare number, left alone
    "redup":      ("GANDA",    False),   # budak2 -> budak
    "elong":      ("PANJANG",  False),   # gilaaa -> gila
    # An English word, detected by a Hunspell en_US lookup. This is the tag
    # that makes the set a ROJAK lemmatiser rather than a Malay one, and it is
    # only emitted when pyenchant is installed -- without it, code-switched
    # words fall to `oov` and the model never learns INGGERIS at all. Install
    # pyenchant before pre-filling any sheet.
    "codeswitch": ("INGGERIS", False),
    "affix":      ("IMBUHAN",  True),    # MALINDO root column; needs checking
    # MALINDO offered several lemmas and one was picked. A real dictionary
    # lemma, but the choice between candidates is the part that needs a human.
    "ambiguous":  ("KAMUS",    True),
    # MALINDO echoed an informal spelling back at us. Tagging that KAMUS would
    # teach the model that informal spellings are roots, so it is KEKAL --
    # kept, not resolved.
    "dict_check": ("KEKAL",    True),
    "oov":        ("KEKAL",    True),    # nothing resolved it
    "unknown":    ("KEKAL",    True),
}

# "human" is not a rule. When an annotator corrects a lemma the original rule
# is overwritten, so WHICH rule fired is gone -- while the lemma itself became
# the most trustworthy label in the project. Dumping those tokens into KEKAL
# would throw away the best data we have, so rule_for() re-derives the rule
# from the surface pair instead. See infer_rule().
HUMAN_SRC = "human"


# ---------------------------------------------------------------------------
# THE SKILL
# ---------------------------------------------------------------------------
#
# Every "JANGAN" below is a failure this pipeline actually produces, not a
# generic caution:
#
#   - translating to English is the base model's default reading of "convert
#     this Malay sentence"
#   - answering the question is what any interrogative input triggers: given
#     "kenapa mahal sangat?" an instruct model explains inflation
#   - one line per token is load-bearing: scoring aligns the answer against
#     the input token list positionally, and a row whose token count drifted
#     scores zero on every token, not just the drifted one
#   - particles are a deliberate do-not-normalise set; "la" is not "lah" is
#     not an error to fix
#   - "dengan" -> "gan" and "makan" -> "kan" are what a naive affix stripper
#     does to roots that merely start with a prefix-shaped string
#   - splitting "budak2" into "budak budak" breaks the alignment AND the label

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

PROMPTS = {"v1-reasoning": SYSTEM_PROMPT}
DEFAULT_VERSION = SKILL_VERSION


def version_for(adapter_dir):
    """Which skill version an adapter expects.

    Read from <adapter>/skill.json, stamped at save time. An unstamped adapter
    falls back to the current version WITH A WARNING, because guessing wrong
    here produces the failure that is hardest to notice: the model still
    answers fluently, the answer is just the wrong format, and the score comes
    out low for a reason that has nothing to do with the model.
    """
    meta = os.path.join(str(adapter_dir), SKILL_FILE)
    if os.path.exists(meta):
        try:
            with open(meta, encoding="utf-8") as fh:
                v = json.load(fh).get("skill_version")
            if v in PROMPTS:
                return v
        except (OSError, ValueError):
            pass          # an unreadable stamp is no worse than no stamp
    print("[skill] %s carries no %s -- assuming %s. Stamp it with "
          "write_version() to make this certain."
          % (adapter_dir, SKILL_FILE, DEFAULT_VERSION))
    return DEFAULT_VERSION


def prompt_for(adapter_dir):
    """(system prompt, version) for an adapter.

    The only supported way to get a system prompt at inference time. Anything
    that keeps its own copy will eventually be served against a model trained
    on a different one.
    """
    v = version_for(adapter_dir)
    return PROMPTS[v], v


def write_version(adapter_dir, version=SKILL_VERSION):
    """Stamp a freshly saved adapter with the skill version it was trained on."""
    if version not in PROMPTS:
        raise ValueError("unknown skill version %r" % (version,))
    path = os.path.join(str(adapter_dir), SKILL_FILE)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"skill_version": version, "skill_name": SKILL_VERSION},
                  fh, indent=2)
    return path


# ---------------------------------------------------------------------------
# RENDER: corpus annotations -> training target
# ---------------------------------------------------------------------------

_REDUP_RE = re.compile(r"^(.+?)2$")
_ELONG_RE = re.compile(r"(.)\1{1,}")


def infer_rule(token, lemma):
    """Re-derive the rule from a token/lemma pair, for labels that lost theirs.

    Only used where an annotator's correction overwrote the rule. Surface
    evidence recovers the mechanical cases exactly; anything left is a change
    a person made for a reason the two strings do not show, so it comes back
    as RINGKAS with the uncertainty marker set -- the lemma is trustworthy,
    the rule label is a guess, and the "?" says so.
    """
    t, l = token.lower(), lemma.lower()
    if t == l:
        return ("KAMUS", False)
    m = _REDUP_RE.match(t)
    if m and m.group(1) == l:
        return ("GANDA", False)
    if "-" in t and t.split("-")[0] == l:
        return ("GANDA", False)
    if _ELONG_RE.sub(r"\1", t) == _ELONG_RE.sub(r"\1", l):
        return ("PANJANG", False)
    if t.endswith(l) or t.startswith(l):
        return ("IMBUHAN", False)          # a prefix or suffix came off
    return ("RINGKAS", True)               # a substitution, reason not visible


def rule_for(src, token=None, lemma=None):
    """(rule tag, uncertain) for one annotated token."""
    if src == HUMAN_SRC and token is not None and lemma is not None:
        return infer_rule(token, lemma)
    return SRC_TO_RULE.get(src, ("KEKAL", True))


def render_target(row):
    """A corpus row -> the exact string the model is trained to produce.

    Takes a row in the corpus schema:
        {"tokens": [...], "lemmas": [{"t": ..., "l": ..., "src": ...}, ...],
         "lemma_text": "..."}

    The lemma sentence is rebuilt from the per-token lemmas rather than copied
    from `lemma_text`, so the <ayat> block can never disagree with the <lema>
    block above it. A model trained on targets that contradict themselves
    learns to contradict itself.
    """
    lines, words = [], []
    for e in row["lemmas"]:
        tag, uncertain = rule_for(e["src"], e["t"], e["l"])
        lines.append("%s | %s%s | %s"
                     % (e["t"], tag, "?" if uncertain else "", e["l"]))
        words.append(e["l"])
    return "<lema>\n%s\n</lema>\n<ayat>%s</ayat>" % ("\n".join(lines),
                                                    " ".join(words))


def to_chatml(row, system_prompt=SYSTEM_PROMPT):
    """A corpus row -> one ChatML training example."""
    return {"messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": row["text"]},
        {"role": "assistant", "content": render_target(row)},
    ]}


# ---------------------------------------------------------------------------
# PARSE: model answer -> structured result
# ---------------------------------------------------------------------------

LEMA_BLOCK_RE = re.compile(r"<lema>(.*?)</lema>", re.S | re.I)
AYAT_BLOCK_RE = re.compile(r"<ayat>(.*?)</ayat>", re.S | re.I)
LINE_RE = re.compile(r"^\s*(\S+)\s*\|\s*([A-Za-z]+)(\??)\s*\|\s*(.+?)\s*$")


def parse_response(text):
    """Parse a model answer into {"steps": [...], "sentence": str, "ok": bool}.

    Each step is {"token", "rule", "uncertain", "lemma"}.

    Deliberately forgiving about everything except the parts that carry
    meaning. A base model with no fine-tuning will produce prose, markdown
    fences, missing tags and invented rule names, and the evaluation needs a
    number for how often that happens rather than an exception. `ok` is False
    when the answer did not arrive in the required shape at all -- that is the
    format-failure rate, and for an un-fine-tuned baseline it is a headline
    result, not an error to suppress.

    An unrecognised rule name is kept in the step but flagged uncertain: the
    lemma may still be right, and throwing the line away would hide a correct
    answer behind a formatting mistake.
    """
    block = LEMA_BLOCK_RE.search(text)
    body = block.group(1) if block else text

    steps = []
    for line in body.splitlines():
        m = LINE_RE.match(line)
        if not m:
            continue
        token, rule, mark, lemma = m.groups()
        rule = rule.upper()
        steps.append({"token": token,
                      "rule": rule if rule in RULES else "KEKAL",
                      "uncertain": bool(mark) or rule not in RULES,
                      "lemma": lemma.strip()})

    ayat = AYAT_BLOCK_RE.search(text)
    if ayat:
        sentence = _collapse_ws(ayat.group(1))
    else:
        sentence = " ".join(s["lemma"] for s in steps)

    return {"steps": steps, "sentence": sentence,
            "ok": bool(block and steps)}


def _collapse_ws(s):
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------------------
# EXPLAIN: tag -> the prose a human reads
# ---------------------------------------------------------------------------
#
# Generated here, never by the model. The model picks a tag; this turns the
# tag into a sentence. That separation is what makes the explanation
# trustworthy: it cannot drift from the operation that was actually performed.

TAGS_EN = {
    "KAMUS":    "dictionary",
    "RINGKAS":  "slang expanded",
    "IMBUHAN":  "affix stripped",
    "GANDA":    "reduplication",
    "PANJANG":  "elongation",
    "PARTIKEL": "particle",
    "INGGERIS": "English",
    "NOMBOR":   "number",
    "KEKAL":    "unchanged",
}


def explain(rule, uncertain=False, lang="ms"):
    """One line of prose for a rule tag."""
    table = RULES if lang == "ms" else RULES_EN
    text = table.get(rule, table["KEKAL"])
    if uncertain:
        text += " (tidak pasti)" if lang == "ms" else " (uncertain)"
    return text


def format_result(parsed, lang="ms"):
    """Render a parsed answer as the final user-facing output.

    This is the last stage of the architecture: input -> preprocess -> model
    -> here. It shows the lemma sentence first, because that is what was
    asked for, then the derivation, because that is what makes it checkable.
    """
    tags = TAGS_EN if lang == "en" else None
    out = [parsed["sentence"], ""]
    width = max((len(s["token"]) for s in parsed["steps"]), default=0)
    for s in parsed["steps"]:
        label = (tags or {}).get(s["rule"], s["rule"])
        out.append("  %-*s  ->  %-12s  %s%s"
                   % (width, s["token"], s["lemma"], label,
                      "?" if s["uncertain"] else ""))
    if not parsed["ok"]:
        out.append("")
        out.append("[format tidak lengkap]" if lang == "ms"
                   else "[malformed answer]")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEMO_RESPONSE = """\
<lema>
budak2 | GANDA | budak
tu | RINGKAS | itu
gilaaa | PANJANG | gila
</lema>
<ayat>budak itu gila</ayat>"""


def _load_rows(path, n):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= n:
                break
    return rows


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--card", action="store_true", help="print the system prompt")
    p.add_argument("--rules", action="store_true", help="print the rule table")
    p.add_argument("--demo", action="store_true",
                   help="parse the built-in answer, or render --seed rows")
    p.add_argument("--parse", metavar="FILE",
                   help="parse a model answer from a file ('-' for stdin)")
    p.add_argument("--seed", metavar="JSONL",
                   help="corpus file to render training targets from")
    p.add_argument("-n", type=int, default=3, help="how many rows (default 3)")
    p.add_argument("--lang", default="ms", choices=("ms", "en"))
    a = p.parse_args()

    if a.card:
        print(SYSTEM_PROMPT)
        return
    if a.rules:
        print("%-9s %-42s %s" % ("TAG", "GLOSS (ms)", "GLOSS (en)"))
        print("-" * 92)
        for tag in RULES:
            print("%-9s %-42s %s" % (tag, RULES[tag], RULES_EN[tag]))
        print("\n%d src values map onto these tags:" % len(SRC_TO_RULE))
        for src, (tag, unc) in sorted(SRC_TO_RULE.items()):
            print("  %-12s -> %-9s%s" % (src, tag, "?" if unc else ""))
        return
    if a.parse:
        text = sys.stdin.read() if a.parse == "-" else \
            open(a.parse, encoding="utf-8").read()
        print(format_result(parse_response(text), a.lang))
        return
    if a.demo:
        if a.seed:
            for row in _load_rows(a.seed, a.n):
                print("INPUT: %s\n" % row["text"])
                print(render_target(row))
                print("\n" + "-" * 70 + "\n")
        else:
            print("MODEL ANSWER\n%s\n\nPARSED\n" % DEMO_RESPONSE)
            print(format_result(parse_response(DEMO_RESPONSE), a.lang))
        return
    p.print_help()


if __name__ == "__main__":
    main()
