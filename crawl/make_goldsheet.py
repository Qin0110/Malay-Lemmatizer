#!/usr/bin/env python
"""
Build (and collect) the gold-standard labelling sheet.

    python crawl/make_goldsheet.py              # build data/gold_sheet.xlsx
    python crawl/make_goldsheet.py --collect    # sheet -> data/gold.jsonl

WHY THIS EXISTS
    Every accuracy number in the report is measured against this file. It is
    the only part of the dataset with human-verified labels: the training set
    is silver (rule-generated), so a model trained on it can only ever be as
    good as the rules unless something independent says otherwise. This is
    that independent thing.

HOW IT WORKS
    You do NOT type lemmas from scratch. Each token is pre-filled by the same
    rules the pipeline uses, and you only correct what is wrong - roughly 3-4
    tokens a sentence rather than 11. The `rule` column says where each guess
    came from, so you know which to distrust:

        particle   kept as-is on purpose (la, weh, kot)   - rarely wrong
        slang      from the SLANG lexicon (dh -> sudah)   - rarely wrong
        dict       MALINDO resolved it cleanly            - rarely wrong
        affix      MALINDO root column (bermain -> main)  - check
        deriv      derived noun kept whole                - CHECK
                   (pengangkutan stays pengangkutan)
        redup      digit reduplication (budak2 -> budak)  - check
        elong      elongation collapsed (gilaaa -> gila)  - check
        num        a bare number, left alone               - never wrong
        ambiguous  MALINDO gave several lemmas, picked 1  - CHECK
        dict_check MALINDO echoed an informal spelling    - CHECK
        oov        nothing resolved it, left unchanged    - CHECK

    Sort by `rule` and work through `oov` and `ambiguous` first. Those are
    where the fine-tune has to beat the baseline, so they are where a wrong
    label does the most damage.

STEMMING IS NOT LEMMATISATION
    MALINDO's root column reduces every affixed form to its root, which for
    Malay's mostly-DERIVATIONAL affixation is stemming: it turns
    "pengangkutan" (transportation) into "angkut" (to carry) and
    "pengalaman" (experience) into "alam" (world). Those are different words
    with their own dictionary entries, so their lemma is themselves.

    The `deriv` rule draws the line. A nominalising affix builds a new lexeme
    and is kept; a verbal affix (me-, ber-, di-, ter-, -kan, -i) marks voice
    on the same lexeme and is stripped. See is_derived_noun().

OUTPUT
    data/gold.jsonl in the same shape as unsloth/seed500.jsonl, so it drops
    straight into llm.py --eval as the held-out test set.
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "unsloth"))

from databuilder import SLANG, PARTICLES        # noqa: E402

GOLD_POOL = ROOT / "data" / "gold_pool.jsonl"
GOLD_SHEET = ROOT / "data" / "gold_sheet.xlsx"
GOLD_JSONL = ROOT / "data" / "gold.jsonl"
MALINDO_TSV = ROOT / "malindo_morph" / "malindo_dic_2024.tsv"

# Order matters: Python alternation takes the FIRST match, so the
# reduplication form must come first or "baik2" splits into "baik"+"2".
WORD_RE = re.compile(r"[a-z]+\d|[a-z]+(?:-[a-z]+)*|\d+")
ELONG_RE = re.compile(r"([a-z])\1{2,}")
DOUBLE_RE = re.compile(r"([a-z])\1")
REDUP_RE = re.compile(r"^([a-z]{3,})2$")


# ---------------------------------------------------------------------------

def malindo_lemmas():
    """surface form (lower) -> list of distinct lemmas."""
    import pandas as pd
    df = pd.read_csv(MALINDO_TSV, sep="\t", header=None, usecols=[2, 9],
                     names=["surface", "lemma"], encoding="utf-8", dtype=str)
    df = df.dropna()
    out = defaultdict(list)
    for surface, lemma in zip(df["surface"].str.lower(), df["lemma"]):
        if lemma not in out[surface]:
            out[surface].append(lemma)
    return out


def malindo_roots():
    """surface form (lower) -> morphological root.

    MALINDO's `root` column is what a lemmatiser actually wants; the `lemma`
    column often just echoes the surface form. Root handles circumfixes that
    naive prefix-stripping cannot (keselamatan -> selamat, percakapan ->
    cakap) and correctly leaves single-morpheme words that merely look
    affixed (mereka, selalu, perempuan, kelakar) alone.

    It is consulted only AFTER the slang lexicon, because MALINDO lists
    informal spellings with themselves as root - going to it first would
    turn "yg" -> "yang" back into "yg" -> "yg".
    """
    import pandas as pd
    df = pd.read_csv(MALINDO_TSV, sep="\t", header=None, usecols=[1, 2],
                     names=["root", "surface"], encoding="utf-8", dtype=str)
    df = df.dropna()
    out = {}
    for surface, root in zip(df["surface"].str.lower(), df["root"].str.lower()):
        out.setdefault(surface, root)
    return out


BM_DICT_CSV = ROOT / "malindo_morph" / "BM_dict.csv"
_BM = None
_EN = None


def english_dict():
    """A Hunspell en_US dictionary, or None if pyenchant is unavailable.

    Optional on purpose: without it the labeller behaves exactly as it did
    before, marking code-switched words `oov` instead of `codeswitch`.
    """
    global _EN
    if _EN is None:
        try:
            import enchant
            _EN = enchant.Dict("en_US")
        except Exception:
            _EN = False
    return _EN or None


def looks_english(token):
    """True for a token an English dictionary accepts.

    Length and alphabetic guards because enchant accepts single letters and
    bare digits, and because a two-letter Malay particle should never be
    diverted here.
    """
    if len(token) < 3 or not token.isalpha():
        return False
    d = english_dict()
    return bool(d and d.check(token))


def bm_dict():
    """rojak -> standard, from BM_dict.csv. First spelling of a key wins.

    Missing file is not fatal: the labeller falls back to SLANG alone, which
    is what it did before this was added.
    """
    global _BM
    if _BM is None:
        import csv
        out = {}
        try:
            with BM_DICT_CSV.open(encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    key = (row.get("rojak") or "").strip().lower()
                    val = (row.get("actual") or "").strip()
                    if key and val:
                        out.setdefault(key, val)
        except OSError:
            pass
        _BM = out
    return _BM


VOWELS = set("aeiou")


def pick(lemmas, token):
    """Choose one lemma from MALINDO's candidates.

    Two wrinkles in the data: a single lemma cell can pack alternatives
    separated by "@" ("bagi@membagi"), and the first candidate is often an
    inflected form ("ikut" -> "mengikut"), which is the opposite of a lemma.
    Prefer the surface form when it is itself a listed lemma, else the
    shortest - the least affixed candidate is the best default.
    """
    cands = []
    for lemma in lemmas:
        for part in str(lemma).split("@"):
            part = part.strip().lower()
            if part and part not in cands:
                cands.append(part)
    if not cands:
        return None, 0
    if token in cands:
        return token, len(cands)
    return min(cands, key=len), len(cands)


def suffix_of(token, root):
    """Which suffix is attached, identified AGAINST THE ROOT.

    Testing the string end alone is wrong in both directions, and both
    failures are common:

        berjalan     ends in "an" but the "an" belongs to the root (jalan),
                     so this is a bare ber- prefix, not a nominalisation
        kecantikan   ends in "kan" but the suffix is -an; the "k" is the last
                     letter of the root (cantik)

    Requiring the root to appear in the remainder settles both. Returns "" when
    no suffix is attached.
    """
    if token.endswith("kan") and root in token[:-3]:
        return "kan"
    if token.endswith("i") and root in token[:-1]:
        return "i"
    if token.endswith("an") and root in token[:-2]:
        return "an"
    return ""


def is_derived_noun(token, root, lex):
    """Is this a NEW lexeme (keep it) rather than a form of the root (strip it)?

    Two things have to hold. The affix must be nominalising -- the -an
    suffix, or a peN-/ke- prefix forming an agent or abstract noun -- and
    MALINDO must list the token as its own headword. The second condition is
    what stops a plausible-looking analysis from inventing a word: if the
    dictionary has no entry for it, it is not a lemma.
    """
    if suffix_of(token, root) in ("kan", "i"):
        return False                        # -kan / -i are verbal, always
    if not headword(token, lex):
        return False
    if suffix_of(token, root) == "an":
        return True                         # pengangkutan, tindakan, bacaan
    return token.startswith(("pe", "ke"))   # pemain, penulis, kekasih


def headword(token, lex):
    """Does MALINDO list this token as its OWN lemma?"""
    if token not in lex:
        return False
    lemma, _ = pick(lex[token], token)
    return lemma == token


def looks_nonstandard(token):
    """Vowel-starved tokens are dropped-vowel rojak, not standard Malay.

    MALINDO contains plenty of informal spellings mapped to themselves
    ("skolah" -> "skolah", "frm" -> "frm"), so a clean dictionary hit is
    not proof the lemma is right. This catches the identity mappings worth
    a second look without flagging ordinary words like "guru" or "dulu".
    """
    letters = [c for c in token if c.isalpha()]
    if len(letters) < 2:
        return True
    return sum(1 for c in letters if c in VOWELS) / len(letters) < 0.35


def propose(token, lex, roots=None):
    """Best guess at the lemma, plus which rule produced it.

    Order matters. Particles come first because several of them (ke, kan,
    pun) are also real words MALINDO would happily inflect. Orthographic
    repair runs before dictionary lookup so "gilaaa" gets a chance to
    resolve as "gila".
    """
    if token.isdigit():
        return token, "num"
    if token in PARTICLES:
        return token, "particle"
    if token in SLANG:
        return SLANG[token], "slang"
    bm = bm_dict()
    if token in bm:
        # Same operation as SLANG, so the same tag. SLANG is checked first,
        # so it keeps the 16 keys where the two lexicons disagree.
        return bm[token], "slang"

    m = REDUP_RE.match(token)
    if m:
        base = m.group(1)
        lemma, _ = pick(lex.get(base, []), base)
        return (lemma or base), "redup"

    # Three or more repeats only. Collapsing TWO repeats was tried and
    # reverted: MALINDO carries Malay-ised spellings of English loanwords,
    # so "all"->"al", "hello"->"helo" and "support"->"suport" all looked
    # like successful resolutions. There is no lexical test that separates
    # Malay elongation from an ordinary English double letter, so tokens
    # like "tolonglahh" are left for the annotator instead.
    collapsed = ELONG_RE.sub(r"\1", token)
    if collapsed != token:
        if collapsed in SLANG:
            return SLANG[collapsed], "elong"
        if collapsed in bm:
            return bm[collapsed], "elong"
        root = roots.get(collapsed) if roots else None
        if root:
            return root, "elong"
        lemma, _ = pick(lex.get(collapsed, []), collapsed)
        return (lemma or collapsed), "elong"

    root = roots.get(token) if roots else None
    if root and root != token:
        # MALINDO gives a ROOT here, not a lemma, and the two are different
        # questions. Reducing every affixed form to its root is stemming:
        # it turns "pengangkutan" (transportation) into "angkut" (to carry)
        # and "pengalaman" (experience) into "alam" (world), which are not
        # the same words. Malay affixation is mostly DERIVATIONAL -- it
        # builds new lexemes with their own dictionary entries -- so the
        # lemma of a derived noun is the derived noun.
        #
        # The line drawn here:
        #   nominalising affixes  -> a new lexeme  -> KEEP  (pengangkutan)
        #   verbal affixes        -> the same lexeme -> STRIP (memakan -> makan)
        #
        # Malay has almost no inflection, so "verbal affix" here means the
        # voice/transitivity markers (me-, ber-, di-, ter-, -kan, -i), which
        # every grammar treats as forms of one verb rather than separate
        # words.
        if is_derived_noun(token, root, lex):
            return token, "deriv"
        # Surfaced as `affix` so the annotator can still spot over-stripping
        # (semoga -> moga), which the root column does produce.
        return root, "affix"

    if token in lex:
        lemma, n = pick(lex[token], token)
        if n > 1:
            return lemma, "ambiguous"
        # A dictionary hit that changes nothing, on a vowel-starved token,
        # is MALINDO echoing an informal spelling back at us.
        if lemma == token and looks_nonstandard(token):
            return lemma, "dict_check"
        return lemma, "dict"

    # Last, so that anything a Malay rule can explain has already been taken.
    # MALINDO lists English borrowings ("the", "real", "margin"), so most
    # common English is caught above as `dict`; what reaches here is the
    # code-switching the Malay side genuinely has no account of.
    if looks_english(token):
        return token, "codeswitch"

    return token, "oov"


# ---------------------------------------------------------------------------

def build():
    if not GOLD_POOL.exists():
        sys.exit("No %s. Run crawl/build_corpus.py first."
                 % GOLD_POOL.relative_to(ROOT))
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill
    except ImportError:
        sys.exit("openpyxl is needed: pip install openpyxl")

    print("loading MALINDO...")
    lex = malindo_lemmas()
    roots = malindo_roots()
    print("%d surface forms, %d roots\n" % (len(lex), len(roots)))

    rows = [json.loads(l) for l in GOLD_POOL.open(encoding="utf-8") if l.strip()]

    prior = _prior_corrections()
    if prior:
        archive = GOLD_SHEET.with_name(
            "gold_sheet_labelled_%s.xlsx" % time.strftime("%Y%m%d_%H%M%S"))
        GOLD_SHEET.rename(archive)
        print("Carrying forward %d corrections from your previous sheet."
              % len(prior))
        print("(previous sheet archived as %s)\n" % archive.name)

    wb = Workbook()
    ws = wb.active
    ws.title = "gold"
    ws.append(["sent_id", "sentence", "token", "proposed_lemma",
               "corrected_lemma", "rule"])

    counts = defaultdict(int)
    n_tokens = 0
    for row in rows:
        tokens = WORD_RE.findall(row["text"].lower())
        for i, tok in enumerate(tokens):
            lemma, rule = propose(tok, lex, roots)
            counts[rule] += 1
            n_tokens += 1
            carried = prior.get((row["id"], tok))
            ws.append([row["id"],
                       row["text"] if i == 0 else "",
                       tok, lemma, carried, rule])

    for cell in ws[1]:
        cell.font = Font(bold=True)
    widths = {"A": 11, "B": 60, "C": 16, "D": 18, "E": 18, "F": 11}
    for col, w in widths.items():
        ws.column_dimensions[col].width = w
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions

    # Tint the rows that actually need judgement.
    warn = PatternFill("solid", fgColor="FFF2CC")
    hot = PatternFill("solid", fgColor="FCE4D6")
    for r in range(2, ws.max_row + 1):
        rule = ws.cell(row=r, column=6).value
        if rule in ("oov", "dict_check"):
            for c in range(1, 7):
                ws.cell(row=r, column=c).fill = hot
        elif rule in ("ambiguous", "affix", "redup", "elong", "deriv"):
            for c in range(1, 7):
                ws.cell(row=r, column=c).fill = warn
    for r in range(2, ws.max_row + 1):
        ws.cell(row=r, column=2).alignment = Alignment(vertical="top")

    wb.save(GOLD_SHEET)

    print("%d sentences, %d tokens -> %s\n"
          % (len(rows), n_tokens, GOLD_SHEET.relative_to(ROOT)))
    print("%-11s %6s %7s" % ("rule", "n", "share"))
    print("-" * 26)
    for rule, n in sorted(counts.items(), key=lambda x: -x[1]):
        flag = ("  <- check these"
                if rule in ("oov", "dict_check", "ambiguous", "deriv") else "")
        print("%-11s %6d %6.1f%%%s" % (rule, n, 100 * n / n_tokens, flag))

    need = sum(counts[k] for k in
               ("oov", "dict_check", "ambiguous", "affix", "redup", "elong",
                "deriv"))
    print("\nRows tinted for attention: %d of %d (%.0f%%)"
          % (need, n_tokens, 100 * need / n_tokens))
    print("Everything else is very likely already correct - skim, do not retype.")
    print("\nFill `corrected_lemma` ONLY where `proposed_lemma` is wrong.")
    print("Leave it blank to accept the proposal.")
    # Echo back the paths this run actually used. A --pool run that printed
    # the bare default would send the annotator's finished work into the gold
    # files, which is the one mistake this whole three-way split exists to
    # prevent.
    if GOLD_POOL == ROOT / "data" / "gold_pool.jsonl":
        print("\nWhen done: python crawl/make_goldsheet.py --collect")
    else:
        print("\nWhen done: python crawl/make_goldsheet.py --collect \\\n"
              "             --pool %s --sheet %s --out %s"
              % (GOLD_POOL.relative_to(ROOT), GOLD_SHEET.relative_to(ROOT),
                 GOLD_JSONL.relative_to(ROOT)))


def _prior_corrections():
    """Corrections already made, keyed by (sent_id, token).

    Rebuilding the sheet used to throw away hand-labelling. Carrying the
    edits forward means a fix to the pre-annotation costs nothing, so there
    is never a reason to keep working against a sheet known to be wrong.
    """
    if not GOLD_SHEET.exists():
        return {}
    try:
        from openpyxl import load_workbook
        ws = load_workbook(GOLD_SHEET, data_only=True).active
    except Exception:
        return {}
    out = {}
    for r in ws.iter_rows(min_row=2, values_only=True):
        if r and len(r) > 4 and r[0] and r[2] and r[4] and str(r[4]).strip():
            out[(r[0], r[2])] = str(r[4]).strip()
    return out


# ---------------------------------------------------------------------------

def _text(v):
    """A spreadsheet cell as the text it was written as.

    Excel stores a cell that looks numeric AS a number, so openpyxl hands back
    int 2 for the token "2" -- and every corpus token that is a bare number
    (counts, years, prices) comes back that way. The old code called .lower()
    on it and died. Worse than dying would have been not dying: json.dumps
    would then have written `"t": 2` into the annotation, and every downstream
    comparison against the string "2" would silently miss.

    A float that is integral is rendered without the ".0", because Excel will
    sometimes store 8 as 8.0 and "8.0" would not match the token in the pool.
    """
    if v is None:
        return ""
    if isinstance(v, bool):          # bool is an int subclass; catch it first
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() else repr(v)
    return str(v).strip()


def collect():
    if not GOLD_SHEET.exists():
        sys.exit("No %s yet." % GOLD_SHEET.relative_to(ROOT))
    from openpyxl import load_workbook

    pool = {json.loads(l)["id"]: json.loads(l)
            for l in GOLD_POOL.open(encoding="utf-8") if l.strip()}
    lex = malindo_lemmas()

    ws = load_workbook(GOLD_SHEET, data_only=True).active
    per_sent, edits = defaultdict(list), 0
    for r in ws.iter_rows(min_row=2, values_only=True):
        if not r or not r[0]:
            continue
        sent_id, _, token, proposed, corrected, rule = (_text(v) for v in r[:6])
        if corrected:
            edits += 1
        per_sent[sent_id].append({
            "t": token,
            "l": (corrected or proposed or token).lower(),
            "src": ("human" if corrected else rule),
        })

    out = []
    for sent_id, lemmas in per_sent.items():
        row = pool.get(sent_id)
        if not row:
            continue
        out.append({
            "id": sent_id,
            "text": row["text"],
            "tokens": [e["t"] for e in lemmas],
            "candidates": {e["t"]: lex.get(e["t"], []) for e in lemmas},
            "lemmas": lemmas,
            "lemma_text": " ".join(e["l"] for e in lemmas),
            "band": row.get("band"),
        })

    with GOLD_JSONL.open("w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_tok = sum(len(r["lemmas"]) for r in out)
    print("%d sentences, %d tokens -> %s"
          % (len(out), n_tok, GOLD_JSONL.relative_to(ROOT)))
    print("%d tokens corrected by hand (%.1f%%)" % (edits, 100 * edits / max(n_tok, 1)))
    src = defaultdict(int)
    for row in out:
        for e in row["lemmas"]:
            src[e["src"]] += 1
    print()
    for k, n in sorted(src.items(), key=lambda x: -x[1]):
        print("  %-11s %5d" % (k, n))
    if edits == 0:
        print("\nNo corrections at all - did you fill the corrected_lemma column?")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--collect", action="store_true",
                   help="read the finished sheet back into the --out file")
    # The same sheet machinery builds the VALIDATION set, which is a different
    # pool going to a different destination. Passing paths rather than copying
    # this file means the two sets are pre-filled by identical rules, which is
    # what makes their scores comparable -- a second copy would drift.
    #
    # All three move together. Overriding --pool while leaving --sheet at its
    # default would silently overwrite the gold sheet with validation rows, so
    # the group is validated as a unit below.
    p.add_argument("--pool", metavar="JSONL",
                   help="sentence pool to label (default data/gold_pool.jsonl)")
    p.add_argument("--sheet", metavar="XLSX",
                   help="labelling sheet to write/read (default data/gold_sheet.xlsx)")
    p.add_argument("--out", metavar="JSONL",
                   help="collected output (default data/gold.jsonl)")
    a = p.parse_args()

    overrides = [a.pool, a.sheet, a.out]
    if any(overrides) and not all(overrides):
        p.error("--pool, --sheet and --out must be given together: overriding "
                "only some of them would write one set's rows into another "
                "set's files.")
    if a.pool:
        # Resolved against ROOT, not the shell's cwd: build() and collect()
        # report their paths with .relative_to(ROOT), which raises on a bare
        # relative path handed in from somewhere else in the tree.
        def _at(p):
            p = Path(p)
            return p if p.is_absolute() else (ROOT / p).resolve()

        GOLD_POOL = _at(a.pool)
        GOLD_SHEET = _at(a.sheet)
        GOLD_JSONL = _at(a.out)
        if not GOLD_POOL.exists():
            sys.exit("No %s. Run crawl/make_valpool.py first." % GOLD_POOL)
        print("pool  %s\nsheet %s\nout   %s\n" % (GOLD_POOL, GOLD_SHEET,
                                                  GOLD_JSONL))

    collect() if a.collect else build()
