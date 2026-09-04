import argparse
import html
import json
import random
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "unsloth"))

SLANG = {
    "dh": "sudah", "dah": "sudah", "udh": "sudah", "sdh": "sudah",
    "tak": "tidak", "x": "tidak", "tk": "tidak", "xde": "tiada",
    "xleh": "tidak boleh", "xda": "tiada", "xkan": "tidakkan",
    "tkpe": "tidak apa", "takpe": "tidak apa", "tp": "tetapi", "tapi": "tetapi",
    "nak": "mahu", "nk": "mahu", "mau": "mahu", "nok": "mahu",
    "tu": "itu", "ni": "ini", "nie": "ini", "tuh": "itu",
    "kat": "di", "kt": "di", "dkt": "dekat",
    "ko": "kamu", "kau": "kamu", "korang": "kamu",
    "die": "dia", "dye": "dia", "sy": "saya", "sya": "saya",
    "npe": "kenapa", "nape": "kenapa", "knp": "kenapa", "kenape": "kenapa",
    "smlm": "semalam", "smlam": "semalam", "esk": "esok",
    "nnti": "nanti", "nti": "nanti", "skrg": "sekarang", "skang": "sekarang",
    "sbb": "sebab", "sbnrnya": "sebenarnya", "jgn": "jangan", "jgk": "juga",
    "dgn": "dengan", "dgr": "dengar", "mcm": "macam", "cmni": "macam ini",
    "tgh": "tengah", "tgk": "tengok", "blh": "boleh", "bleh": "boleh",
    "keje": "kerja", "kerje": "kerja", "kje": "kerja",
    "umah": "rumah", "umh": "rumah", "bawak": "bawa", "ambik": "ambil",
    "kejap": "sekejap", "jap": "sekejap", "cuba": "cuba", "cube": "cuba",
    "byk": "banyak", "bnyk": "banyak", "sket": "sedikit", "sikit": "sedikit",
    "org": "orang", "org2": "orang", "bkn": "bukan", "bknnya": "bukannya",
    "sume": "semua", "smua": "semua", "sma": "semua",
    "hrga": "harga", "duit": "duit", "wang": "wang",
    "dun": "jangan", "kaw": "kaw", "kow": "kaw",
    "pg": "pergi",
    "blk": "balik", "blike": "balik", "naek": "naik", "turon": "turun",
    "jadik": "jadi", "sgt": "sangat",
    "mmg": "memang", "mesti": "mesti", "msti": "mesti",
    "camne": "bagaimana", "cane": "bagaimana", "mcmne": "bagaimana",
    "bler": "bila", "bile": "bila", "kalo": "kalau", "klu": "kalau",
    "utk": "untuk", "utuk": "untuk", "yg": "yang", "yng": "yang",
    "adalah": "adalah", "ade": "ada", "ada": "ada",
    "gile": "gila", "giler": "gila", "gempak": "gempak",
    "besh": "best", "bestt": "best",
    "pulak": "pula", "pown": "pun",
    "cite": "cerita", "citer": "cerita", "crita": "cerita", "cerite": "cerita",
    "tgok": "tengok", "tengok": "tengok",
    "tq": "terima kasih",
    "lg": "lagi",                # 4x
    "kak": "kakak",              # 5x
    "tpi": "tetapi",             # 4x
    "kiteorg": "kami",           # 2x
    "jd": "jadi",                # 2x
    "betol": "betul",            # 2x
    "tngok": "tengok",           # 2x
    "tgn": "tangan",             # 1x
    "xjadi": "tidak jadi",       # 1x
    "xtau": "tidak tahu",        # 1x
    "xbleh": "tidak boleh",      # 1x
    "xley": "tidak boleh",       # 1x
    "xske": "tidak suka",        # 1x
    "xyh": "tidak payah",        # 1x - gold says "tak payah"; "tidak" wins
}          

# Discourse particles - deliberately NOT normalised
PARTICLES = {"la", "lah", "le", "lor", "leh", "meh", "weh", "wei", "woi",
             "je", "jer", "kan", "kot", "eh", "ah", "ha", "hah", "ke", "kah",
             "pun", "tau", "doh", "bah", "yo", "sia",
             # from the YouTube crawl probe (Aug 2026)
             "beb", "siak", "wey", "weyh", "bro", "bre"}

RAW_FILE = ROOT / "data" / "raw" / "youtube_raw.jsonl"
CORPUS_FILE = ROOT / "data" / "corpus.jsonl"
GOLD_POOL_FILE = ROOT / "data" / "gold_pool.jsonl"
MALINDO_TSV = ROOT / "malindo_morph" / "malindo_dic_2024.tsv"
CHECK_FILE = ROOT / "data" / "filter_check.tsv"
CHECK_XLSX = ROOT / "data" / "filter_check.xlsx"

MIN_WORDS, MAX_WORDS = 3, 30        # matches seed500: median 9, p95 19, max 26
MIN_MALAY_HITS = 2
MIN_ROJAK_HITS = 2
GOLD_POOL_SIZE = 150
SEED = 3407
CHECK_SEED = SEED
FORCE_CHECK = False

# Layer 2: closed-class Malay words. Highest-frequency, most reliable signal
# that a sentence is Malay at all.
MALAY_FUNCTION = {
    "yang", "dan", "di", "ke", "dari", "untuk", "dengan", "ini", "itu",
    "tidak", "ada", "saya", "dia", "kita", "kami", "mereka", "sudah", "akan",
    "juga", "pada", "adalah", "boleh", "nak", "tak", "ni", "tu", "aku", "kau",
    "apa", "kenapa", "macam", "sebab", "kalau", "bila", "semua", "orang",
    "buat", "jangan", "lagi", "sangat", "memang", "bukan", "kena", "dalam",
}

# Layer 3: English markers, for detecting code-switching inside Malay text.
ENGLISH_MARKERS = {
    "the", "is", "are", "you", "your", "and", "but", "with", "have", "has",
    "this", "that", "for", "not", "can", "will", "just", "like", "get", "all",
    "what", "why", "how", "very", "really", "good", "bad", "best", "so",
    "people", "make", "want", "need", "know", "think", "time", "much",
}

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@[\w.-]+")
TIMESTAMP_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
# Runs of dots are emphasis in rojak ("sudah2 lh tu.. jgn buat"), not sentence
# boundaries. Splitting on them shreds sentences into fragments too short to
# carry any rojak evidence, which silently biases the corpus toward long rows.
ELLIPSIS_RE = re.compile(r"\.{2,}")
SENT_SPLIT_RE = re.compile(r"[.!?\n]+|\s{3,}")
# Order matters: Python alternation takes the FIRST match, so the
# reduplication form must come first or "baik2" splits into "baik"+"2".
WORD_RE = re.compile(r"[a-z]+\d|[a-z]+(?:-[a-z]+)*|\d+")

X_NEGATION_RE = re.compile(r"\bx[a-z]{2,}\b")
REDUP_DIGIT_RE = re.compile(r"\b[a-z]{3,}2\b")
ELONGATION_RE = re.compile(r"([a-z])\1{2,}")
NO_VOWEL_RE = re.compile(r"\b(?![0-9])[bcdfghjklmnpqrstvwxyz]{3,}\b")
E_FOR_A_RE = re.compile(
    r"\b(ape|mane|kenape|sape|bile|camne|mcmne|ade|jadinye|kite|nape|"
    r"cane|gile|die|dye|kerje|keje|sume|bende|kejap)\b")


# ---------------------------------------------------------------------------
# CLEANING AND SPLITTING
# ---------------------------------------------------------------------------

def clean(text):
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = TIMESTAMP_RE.sub(" ", text)
    text = ELLIPSIS_RE.sub(" ", text)
    # Emoji, symbols and stray control chars become a SPACE, never nothing:
    # deleting them welds the neighbours together ("best\naku" -> "bestaku").
    # Newlines are kept - in YouTube comments a line break is usually the only
    # sentence boundary there is, so sentences() needs to see them.
    text = "".join(
        c if (c == "\n" or unicodedata.category(c)[0] not in ("S", "C"))
        else " "
        for c in text
    )
    text = re.sub(r"[^\S\n]+", " ", text)          # collapse spaces, keep \n
    return "\n".join(ln.strip() for ln in text.split("\n")).strip()


def sentences(text):
    """Split a comment into candidate sentences. Rojak often has no full
    stops at all, so a long unpunctuated comment stays one chunk and gets
    dropped by the length filter rather than being mangled."""
    for chunk in SENT_SPLIT_RE.split(text):
        chunk = " ".join(chunk.split())
        if chunk:
            yield chunk


def latin_ratio(text):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    latin = sum(1 for c in letters if "LATIN" in unicodedata.name(c, ""))
    return latin / len(letters)


# ---------------------------------------------------------------------------
# ROJAK SCORING
# ---------------------------------------------------------------------------

_SURFACE = None


def malindo_surfaces():
    """Every surface form MALINDO Morph knows, lowercased, as a set.

    Read straight from the TSV rather than through malindo_morph.Dictionary:
    that class does a full-column pandas scan per word (far too slow for
    ~120k lookups) and its morphological_analysis() reports a miss as the
    literal string "Series([], )" rather than an empty value.
    """
    global _SURFACE
    if _SURFACE is None:
        import pandas as pd
        df = pd.read_csv(MALINDO_TSV, sep="\t", usecols=[2], names=["surface"],
                         header=None, encoding="utf-8", dtype=str)
        _SURFACE = set(df["surface"].dropna().str.lower())
    return _SURFACE


def features(sentence, use_malindo=True):
    low = sentence.lower()
    words = WORD_RE.findall(low)
    wordset = set(words)

    f = {
        "slang": sum(1 for w in words if w in SLANG),
        "particle": sum(1 for w in words if w in PARTICLES),
        "x_neg": len(X_NEGATION_RE.findall(low)),
        "redup": len(REDUP_DIGIT_RE.findall(low)),
        "elong": len(ELONGATION_RE.findall(low)),
        "no_vowel": len(NO_VOWEL_RE.findall(low)),
        "e_for_a": len(E_FOR_A_RE.findall(low)),
    }
    english_hits = len(wordset & ENGLISH_MARKERS)

    # "Is this Malay at all?" Standard function words alone are not enough:
    # the most rojak sentences replace every one of them ("mmg", "saye",
    # "xske"), so counting only MALAY_FUNCTION rejects our best examples.
    # Slang, particles, x-negation and e-for-a are Malay-specific lexical
    # evidence too. Orthographic markers (elongation, vowel-dropping,
    # reduplication) are deliberately excluded - they are language-agnostic
    # and would let "soooo goooood" through as Malay.
    malay_hits = (len(wordset & MALAY_FUNCTION)
                  + f["slang"] + f["particle"] + f["e_for_a"] + f["x_neg"])

    # Code-switching only counts when BOTH languages are present.
    f["codeswitch"] = english_hits if (malay_hits >= 2 and english_hits) else 0

    # Tokens MALINDO Morph cannot resolve. Measured against 150 hand-labelled
    # sentences this is the single strongest signal, and it rises cleanly with
    # the count (0 OOV -> 35% rojak, 1 -> 55%, 2 -> 72%, 3+ -> 83%). It is also
    # the only marker that needs no lexicon of ours: "the standard dictionary
    # fails here" is what rojak means for this project.
    oov = 0
    if use_malindo:
        surfaces = malindo_surfaces()
        oov = sum(1 for w in words if w not in surfaces)

    return f, malay_hits, len(words), oov


def difficulty(hits, oov):
    """OOV tokens count double - they are what the fine-tune has to fix."""
    return oov * 2 + hits


def band(hits, oov=0):
    score = difficulty(hits, oov)
    if score >= 10:
        return "high"
    if score >= 6:
        return "mid"
    return "low"


def keeps(hits, malay_hits, n_words, min_hits, oov=0):
    """Keep a sentence if the dictionary struggles with it, or the lexicon
    markers are piled high enough to say so on their own.

    Chosen by measuring candidate rules against 150 hand-labelled sentences.
    This one scored 82% precision / 67% recall, against 70/78 for a flat
    `hits >= 2`. Precision is the one to buy: there are ~13k sentences
    available and only ~3k are needed, so discarding good rows is cheap
    while standard Malay leaking into the training set is not.
    """
    if oov >= 2 or hits >= 4:
        return True
    # Short sentences have no room to accumulate markers, so let one OOV
    # token plus any other evidence carry them.
    if n_words <= 6 and oov >= 1 and hits >= max(1, min_hits - 1):
        return True
    return False


# ---------------------------------------------------------------------------
# BUILD
# ---------------------------------------------------------------------------

def build(min_hits, validate, use_malindo=True):
    if not RAW_FILE.exists():
        sys.exit("No %s yet. Run crawl/youtube_crawl.py first."
                 % RAW_FILE.relative_to(ROOT))

    raw = [json.loads(l) for l in RAW_FILE.open(encoding="utf-8") if l.strip()]
    print("%d raw comments\n" % len(raw))

    kept, rejected, seen = [], [], set()
    drop = Counter()

    for comment in raw:
        for sent in sentences(clean(comment["text"])):
            n_words = len(sent.split())
            if not (MIN_WORDS <= n_words <= MAX_WORDS):
                drop["length"] += 1
                continue

            key = re.sub(r"[^a-z0-9 ]", "", sent.lower())
            key = " ".join(key.split())
            if not key or key in seen:
                drop["duplicate"] += 1
                continue
            seen.add(key)

            if latin_ratio(sent) < 0.8:
                drop["script"] += 1
                continue

            f, malay_hits, n_tokens, oov = features(sent, use_malindo)
            hits = sum(f.values())

            # Layer 2 is length-aware too, so short rojak is not thrown out
            # for lacking two standard function words it has no room for.
            if malay_hits < (1 if n_words <= 6 else MIN_MALAY_HITS):
                drop["not_malay"] += 1
                continue

            row = {
                "id": "yt_%05d" % len(kept),
                "text": sent,
                "video_id": comment["video_id"],
                "n_words": n_words,
                "rojak_hits": hits,
                "oov": oov,
                "difficulty": difficulty(hits, oov),
                "rojak_density": round(hits / max(n_tokens, 1), 3),
                "band": band(hits, oov),
                "features": f,
            }

            if not keeps(hits, malay_hits, n_words, min_hits, oov):
                drop["not_rojak"] += 1
                if validate:
                    rejected.append(row)
                continue
            kept.append(row)

    _report(raw, kept, drop)
    _write(kept)
    if validate:
        _write_check(kept, rejected)


def _write(kept):
    with CORPUS_FILE.open("w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Stratified gold pool so the test set is not accidentally all-easy.
    rng = random.Random(SEED)
    by_band = {"low": [], "mid": [], "high": []}
    for row in kept:
        by_band[row["band"]].append(row)

    pool, per_band = [], GOLD_POOL_SIZE // 3
    for name in ("low", "mid", "high"):
        rows = by_band[name]
        pool += rng.sample(rows, min(per_band, len(rows)))
    # Top up from the largest band if some band was short.
    if len(pool) < GOLD_POOL_SIZE:
        rest = [r for r in kept if r not in pool]
        pool += rng.sample(rest, min(GOLD_POOL_SIZE - len(pool), len(rest)))
    rng.shuffle(pool)

    with GOLD_POOL_FILE.open("w", encoding="utf-8") as f:
        for row in pool:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\nWrote %d sentences  -> data/corpus.jsonl" % len(kept))
    print("Wrote %d gold pool   -> data/gold_pool.jsonl" % len(pool))


def _existing_labels():
    """How many Y/N labels are already in the check workbook.

    Hand labelling costs half an hour, so refuse to silently overwrite it.
    """
    if not CHECK_XLSX.exists():
        return 0
    try:
        from openpyxl import load_workbook
        ws = load_workbook(CHECK_XLSX, data_only=True).active
        return sum(1 for r in ws.iter_rows(min_row=2, values_only=True)
                   if r and r[0] and r[2] and str(r[2]).strip())
    except Exception:
        return 0


def _write_check(kept, rejected):
    """Sample for step 3: hand-check the filter and get precision/recall."""
    n_labels = _existing_labels()
    if n_labels and not FORCE_CHECK:
        archive = CHECK_XLSX.with_name(
            "filter_check_labelled_%s.xlsx" % time.strftime("%Y%m%d_%H%M%S"))
        CHECK_XLSX.rename(archive)
        print("\n  NOTE: %s already held %d labels."
              % (CHECK_XLSX.name, n_labels))
        print("        Moved to %s rather than overwriting it." % archive.name)

    rng = random.Random(CHECK_SEED)
    sample = ([("KEPT", r) for r in rng.sample(kept, min(100, len(kept)))]
              + [("REJECTED", r) for r in rng.sample(rejected, min(50, len(rejected)))])
    rng.shuffle(sample)

    with CHECK_FILE.open("w", encoding="utf-8") as f:
        f.write("filter_said\tis_really_rojak_Y_or_N\thits\ttext\n")
        for verdict, row in sample:
            f.write("%s\t\t%d\t%s\n" % (verdict, row["rojak_hits"], row["text"]))

    print("Wrote %d rows        -> data/filter_check.tsv" % len(sample))
    if _write_xlsx(sample):
        print("Wrote %d rows        -> data/filter_check.xlsx  <- label in this one"
              % len(sample))
    print("\n  Fill the Y/N column, then: python crawl/score_filter.py")


def _write_xlsx(sample):
    """Same sample as an .xlsx, because .tsv does not open cleanly in Excel.

    The filter verdict and hit count are written but hidden: seeing them
    while labelling biases the annotator into agreeing with the filter,
    which is exactly what this sample exists to test. score_filter.py
    still reads them.
    """
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font
        from openpyxl.worksheet.datavalidation import DataValidation
    except ImportError:
        return False

    wb = Workbook()
    ws = wb.active
    ws.title = "filter check"
    ws.append(["filter_said", "hits", "is_really_rojak_Y_or_N", "text"])
    for verdict, row in sample:
        ws.append([verdict, row["rojak_hits"], None, row["text"]])

    for cell in ws[1]:
        cell.font = Font(bold=True)
    ws.column_dimensions["A"].hidden = True      # verdict - hidden, see above
    ws.column_dimensions["B"].hidden = True      # hit count - also a giveaway
    ws.column_dimensions["C"].width = 22
    ws.column_dimensions["D"].width = 110
    for row in ws.iter_rows(min_row=2, min_col=4, max_col=4):
        row[0].alignment = Alignment(wrap_text=False)
    ws.freeze_panes = "A2"

    dv = DataValidation(type="list", formula1='"Y,N"', allow_blank=True,
                        showDropDown=False)
    dv.prompt = "Y if a standard Malay dictionary would struggle, else N"
    dv.promptTitle = "Is this rojak?"
    ws.add_data_validation(dv)
    dv.add("C2:C%d" % (len(sample) + 1))

    wb.save(CHECK_XLSX)
    return True


def _report(raw, kept, drop):
    total_dropped = sum(drop.values())
    print("%-14s %7s" % ("stage", "dropped"))
    print("-" * 24)
    for reason in ("length", "duplicate", "script", "not_malay", "not_rojak"):
        print("%-14s %7d" % (reason, drop[reason]))
    print("-" * 24)
    print("%-14s %7d" % ("total dropped", total_dropped))
    print("%-14s %7d" % ("KEPT", len(kept)))

    if not kept:
        print("\nNothing kept. Lower --min-hits, or crawl channels with more rojak.")
        return

    bands = Counter(r["band"] for r in kept)
    print("\nband      n")
    for name in ("low", "mid", "high"):
        print("  %-6s %6d" % (name, bands[name]))

    feat = Counter()
    for row in kept:
        for k, v in row["features"].items():
            feat[k] += v
    print("\nmarker hits across corpus")
    for k, v in feat.most_common():
        print("  %-12s %6d" % (k, v))

    lens = sorted(r["n_words"] for r in kept)
    print("\nwords/sentence: median %d, p95 %d"
          % (lens[len(lens) // 2], lens[int(len(lens) * 0.95)]))

    print("\nsamples:")
    for row in kept[:5]:
        text = row["text"].encode("ascii", "replace").decode("ascii")
        print("  [%s %2d] %s" % (row["band"], row["rojak_hits"], text[:80]))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--min-hits", type=int, default=MIN_ROJAK_HITS,
                   help="rojak marker hits needed to keep a sentence (default 2)")
    p.add_argument("--validate", action="store_true",
                   help="also write a fresh hand-check sample")
    p.add_argument("--no-malindo", action="store_true",
                   help="skip the dictionary OOV check (faster, much worse)")
    p.add_argument("--force-check", action="store_true",
                   help="overwrite an already-labelled filter_check.xlsx "
                        "instead of archiving it first")
    p.add_argument("--check-seed", type=int, default=SEED,
                   help="seed for the --validate sample; change it to draw a "
                        "FRESH sample after retuning the filter")
    a = p.parse_args()
    CHECK_SEED = a.check_seed
    FORCE_CHECK = a.force_check
    build(a.min_hits, a.validate, not a.no_malindo)
