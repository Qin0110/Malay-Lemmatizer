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


WORD_RE = re.compile(r"[a-z]+\d|[a-z]+(?:-[a-z]+)*|\d+")
ELONG_RE = re.compile(r"([a-z])\1{2,}")
DOUBLE_RE = re.compile(r"([a-z])\1")
REDUP_RE = re.compile(r"^([a-z]{3,})2$")



def malindo_lemmas():
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
    global _EN
    if _EN is None:
        try:
            import enchant
            _EN = enchant.Dict("en_US")
        except Exception:
            _EN = False
    return _EN or None


def looks_english(token):
    if len(token) < 3 or not token.isalpha():
        return False
    d = english_dict()
    return bool(d and d.check(token))


def bm_dict():
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
    if token.endswith("kan") and root in token[:-3]:
        return "kan"
    if token.endswith("i") and root in token[:-1]:
        return "i"
    if token.endswith("an") and root in token[:-2]:
        return "an"
    return ""


def is_derived_noun(token, root, lex):
    if suffix_of(token, root) in ("kan", "i"):
        return False                        # -kan / -i are verbal, always
    if not headword(token, lex):
        return False
    if suffix_of(token, root) == "an":
        return True                         # pengangkutan, tindakan, bacaan
    return token.startswith(("pe", "ke"))   # pemain, penulis, kekasih


def headword(token, lex):
    if token not in lex:
        return False
    lemma, _ = pick(lex[token], token)
    return lemma == token


def looks_nonstandard(token):
    letters = [c for c in token if c.isalpha()]
    if len(letters) < 2:
        return True
    return sum(1 for c in letters if c in VOWELS) / len(letters) < 0.35


def propose(token, lex, roots=None):
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
        if is_derived_noun(token, root, lex):
            return token, "deriv"
        return root, "affix"

    if token in lex:
        lemma, n = pick(lex[token], token)
        if n > 1:
            return lemma, "ambiguous"
        if lemma == token and looks_nonstandard(token):
            return lemma, "dict_check"
        return lemma, "dict"

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

    if GOLD_POOL == ROOT / "data" / "gold_pool.jsonl":
        print("\nWhen done: python crawl/make_goldsheet.py --collect")
    else:
        print("\nWhen done: python crawl/make_goldsheet.py --collect \\\n"
              "             --pool %s --sheet %s --out %s"
              % (GOLD_POOL.relative_to(ROOT), GOLD_SHEET.relative_to(ROOT),
                 GOLD_JSONL.relative_to(ROOT)))


def _prior_corrections():
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
