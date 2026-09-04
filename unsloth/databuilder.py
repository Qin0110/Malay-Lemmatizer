#!/usr/bin/env python
"""
Expand the lemmatizer corpus to 500 rows using the sentiment datasets.

Output target changes from `token -> lemma` lines to a full LEMMA SENTENCE.

    python build_500.py

Produces:
    seed500.jsonl        annotations + per-token provenance (for evaluation)
    train_chatml.jsonl   ChatML, assistant = lemmatised sentence
    review_queue.tsv     tokens needing human verification, most frequent first

IMPORTANT: labels here are RULE-GENERATED (silver), not gold. Verify
review_queue.tsv before treating any of this as ground truth.
"""
import os
import json
import re
import unicodedata
from collections import Counter, defaultdict

# Resolved against __file__, like GANDA_SEMU_FILE below: skill.py induces
# vocabulary from these at inference time, and it can be invoked from any
# directory. A cwd-relative path would silently induce nothing instead.
UPLOADS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sentidata")
MIX = ["mix.train", "mix.valid", "mix.test"]
TSV = ["train_labeled.tsv", "valid_labeled.tsv", "test_labeled.tsv"]
TARGET_ROWS = 500

# ---------------------------------------------------------------------------
# LEXICONS
# ---------------------------------------------------------------------------

# Rojak abbreviations / slang -> standard form
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
    # Added from YouTube crawl probe (Aug 2026) - seen in real comments but
    # absent from the original hand-built lexicon.
    "pulak": "pula", "pown": "pun",
    "cite": "cerita", "citer": "cerita", "crita": "cerita", "cerite": "cerita",
    "tgok": "tengok", "tengok": "tengok",
    "tq": "terima kasih",

    # Promoted from the hand-labelled gold set (data/gold.jsonl, Aug 2026).
    # Every entry here is a decision an annotator actually made on a real
    # sentence - counts are occurrences in the 150 gold sentences. Keeping
    # the lexicon and the gold labels in agreement matters: the silver
    # training data is generated from these rules, so a disagreement would
    # penalise the model for matching its own training signal.
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
}                                #      for consistency with "tak" -> "tidak"

# Discourse particles - deliberately NOT normalised
PARTICLES = {"la", "lah", "le", "lor", "leh", "meh", "weh", "wei", "woi",
             "je", "jer", "kan", "kot", "eh", "ah", "ha", "hah", "ke", "kah",
             "pun", "tau", "doh", "bah", "yo", "sia",
             # from the YouTube crawl probe (Aug 2026)
             "beb", "siak", "wey", "weyh", "bro", "bre"}

# Common Malay roots. Replace with MALINDO Morph output for real work.
ROOTS = set("""
ada ajar alam alih ambil ampun anak angkat api asal atur awal ayah baca bagi
bahagi baik baju baru batu bawa bayar beli benar bengkak berat beri bersih
besar bina buah buang buat buka bunuh bunyi cakap cantik cari cepat cerita
cinta cuba cuci cukup dapat dapur darat datang dengar diam didik dingin duduk
dukung fikir gaji ganti gembira gerak guna hantar harap harga hasil hidup
hilang hitung hormat hubung hujan hukum ikat ikut ingat isi jadi jaga jahat
jalan jauh jawab jelas jual jumpa jual kaji kata kawal kawan kecil kedai kejar
kena kenal kerja kirim kongsi kotor kuat kumpul kuasa lahir lambat lapor larang
lari lawan lengkap letak lihat lupa maaf main makan malu mandi masak masuk mati
minta minum muat mula nama naik nikah nilai nyanyi paham pakai panas pandang
pandu panjang pasang pecah pegang pikir pilih pindah pinjam pukul putus raja
rasa ringan rosak sakit salah sambut sampai sedar sedia sekat sekolah sembah
senang sedap serah simpan siap siasat sihat sokong suka sukat sunyi surat susah
susun tahu tahan tambah tanam tangis tanya tarik tegur tekan telefon tempat
temu tengok terang terima tidur timbang tinggal tinggi tolak tolong tuju tukar
tulis tunggu tunjuk turun tutup ubah ucap ukur undang untung urus usaha wajib
wang warna yakin bangun bantah bantu batal bawah bekal belanja bendung bidang
bilang cadang campur capai catat cerdas darjah dasar edar erti fahami gabung
gagal galak gambar gantung gubah hadap hakim halang hambat hasrat hendak hidang
hindar hitam hujung jamin janji jangka jarak jimat kagum kembang kemas khidmat
kira kubur kurang lantik lapang lawat lebih lindung luar lulus mampu manfaat
milik nampak nasihat niaga nyata olah padan pandai patuh percaya perlu pesan
proses puas rancang rawat rebut rekod ringkas rujuk runding sedia selamat
selesai sempit sengaja serang sertai sesuai sibuk sifat sokong sumbang syak
tadbir taklim tampil tanda tanggung tapis tempah tempuh tentang tentu terbit
terjemah ternak tetap tiba tiru tuntut ulang undur unjur wakil warta wujud
saya anda awak kami kalian beliau ia nya
yang dan atau tetapi kerana jika jikalau kalau maka lalu serta sambil
ini itu sini situ sana mana siapa apa bila mengapa bagaimana berapa
di ke dari pada kepada daripada dalam luar atas bawah antara hingga sehingga
untuk dengan tanpa oleh sejak selain kecuali menurut terhadap mengenai
tidak bukan belum jangan tak
adalah ialah akan sudah telah sedang masih pernah lagi juga pula pun sahaja
sangat amat terlalu paling lebih kurang cukup hampir agak begitu
semua setiap segala beberapa banyak sedikit ramai satu dua tiga
boleh dapat mampu mesti perlu harus patut mahu hendak suka
saja hanya memang tentu pasti mungkin barangkali
ada tiada wujud jadi menjadi ialah
alat anda hebat kes bau nama masa hari malam pagi petang tahun bulan minggu
""".split())

# Vocabulary induced from the corpus itself (populated by induce_vocab()).
# Lines with zero Malay markers give English evidence; Malay-dominant lines give
# Malay evidence. Tokens seen overwhelmingly in one register are assigned to it.
EN_VOCAB = set()
MS_VOCAB = set()

EN_HINT = set("""
the a an and or but if is are was were be been being have has had do does did
you your yours my me we our they them their he she it this that these those
not no yes very much many good bad great best worst nice
""".split())

VOWELS = set("aeiou")

# Roots whose first syllable looks like an affix. Never strip these.
NEVER_STRIP = set("""
dengan jangan bukan makan minum bulan jalan tahun lawan kawan bahan
ikan hutan badan tangan kanan bidan hairan sopan bosan pesan lisan
kerana kerani keranda ketika kepada kecil kereta kedai keluar keluarga kena
kenal kembali kemudian kerap kertas kesan ketua kelas kelmarin kemas kempen
sebab sebelum selepas sedang sedap sedia sekarang sekolah selalu selamat
seluruh semua sentiasa serta sesuatu setiap sebenarnya sekali sementara
memang merah mereka meja mesti melainkan menteri mewah membeli
perlu pergi penuh perang perkara peluang pernah perempuan pertama peluh
berat berapa beras berani bersih besar betul berita
terus terima terang terbaik terkini teruk
dia diam diri dinding dingin
""".split())


# ---------------------------------------------------------------------------
# RULE-BASED LEMMATISER
# ---------------------------------------------------------------------------

def _try(cand):
    return cand if cand in ROOTS else None


def strip_affix(w):
    """Crude Malay affix stripper. Returns (root, matched_rule) or (None, None)."""
    # suffixes first
    base = w
    for suf in ("kannya", "annya", "inya", "nya", "kan", "an", "i"):
        if base.endswith(suf) and len(base) - len(suf) >= 3:
            trimmed = base[: -len(suf)]
            if _try(trimmed):
                return trimmed, f"-{suf}"
            base_alt = trimmed
            break
    else:
        base_alt = base

    for cand in (base_alt, base):
        # simple prefixes
        for pre in ("memper", "member", "diper", "keter", "ber", "ter", "per",
                    "di", "ke", "se"):
            if cand.startswith(pre) and len(cand) - len(pre) >= 3:
                r = _try(cand[len(pre):])
                if r:
                    return r, f"{pre}-"
                # ber- + r-initial doubling: berlari -> lari
                if pre == "ber" and _try("r" + cand[len(pre):]):
                    return "r" + cand[len(pre):], "ber-"

        # nasal prefixes meN- / peN- with sound reversal
        for pre, repl in (
            ("meng", ["k", "", "g", "h"]), ("meny", ["s"]),
            ("mem", ["p", "", "b"]), ("men", ["t", "", "d"]), ("me", [""]),
            ("peng", ["k", "", "g", "h"]), ("peny", ["s"]),
            ("pem", ["p", "", "b"]), ("pen", ["t", "", "d"]), ("pe", [""]),
        ):
            if cand.startswith(pre) and len(cand) - len(pre) >= 3:
                stem = cand[len(pre):]
                for ch in repl:
                    r = _try(ch + stem)
                    if r:
                        return r, f"{pre}-"
    return None, None


def strip_affix_loose(w):
    """Affix stripping that does NOT require the result to be a known root.
    Lower precision, so results are flagged affix_guess for review."""
    if w in NEVER_STRIP:
        return None, False
    base = w
    for suf in ("kannya", "annya", "nya", "kan", "an"):
        if base.endswith(suf) and len(base) - len(suf) >= 4:
            base = base[: -len(suf)]
            break
    # Only unambiguous prefixes. ke-, se-, me-, pe- are excluded: they eat the
    # first syllable of roots like kerana, sebab, memang, perlu.
    for pre, repl in (("meng", "k"), ("meny", "s"), ("mem", "p"), ("men", "t"),
                      ("peng", "k"), ("peny", "s"), ("pem", "p"), ("pen", "t"),
                      ("ber", ""), ("ter", ""), ("di", "")):
        if base.startswith(pre) and len(base) - len(pre) >= 3:
            stem = base[len(pre):]
            # restore the elided initial consonant when the stem starts a vowel
            if repl and stem and stem[0] in VOWELS:
                stem = repl + stem
            base = stem
            break
    return (base, True) if base != w and len(base) >= 3 else (None, False)


def induce_vocab(en_lines, ms_lines):
    """Assign ambiguous tokens to a language using corpus co-occurrence."""
    en_c, ms_c = Counter(), Counter()
    for s in en_lines:
        en_c.update(t.lower() for t in tokenize(s))
    for s in ms_lines:
        ms_c.update(t.lower() for t in tokenize(s))
    for w in set(en_c) | set(ms_c):
        e, m = en_c[w], ms_c[w]
        if e + m < 2:
            continue
        if e >= 2 * max(m, 1) and e >= 2:
            EN_VOCAB.add(w)
        elif m >= 2 * max(e, 1) and m >= 2:
            MS_VOCAB.add(w)


def normalise_repeat(w):
    """gilaa -> gila, sedappp -> sedap. Returns None if nothing collapsed."""
    c = re.sub(r"(.)\1{1,}$", r"\1", w)
    c = re.sub(r"(.)\1{2,}", r"\1", c)
    return c if c != w else None


NUM_RE = re.compile(r"^\d+([.,]\d+)?$")
MONEY_RE = re.compile(r"^rm\s?(\d+(?:[.,]\d+)?)$", re.I)
TIME_RE = re.compile(r"^(\d{1,2})(?:[.:](\d{2}))?\s?(am|pm)$", re.I)
DATE_RE = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{2,4})$")
PCT_RE = re.compile(r"^(\d+(?:\.\d+)?)%$")


def lemma_of(tok):
    """Return (lemma, lang, src). src flags what needs human review."""
    w = tok.lower()

    if MONEY_RE.match(w):
        return f"MYR {MONEY_RE.match(w).group(1)}", "num", "money"
    m = TIME_RE.match(w)
    if m:
        h, mi, ap = int(m.group(1)), m.group(2) or "00", m.group(3).lower()
        if ap == "pm" and h != 12:
            h += 12
        if ap == "am" and h == 12:
            h = 0
        return f"{h:02d}:{mi}", "num", "time"
    m = DATE_RE.match(w)
    if m:
        d, mo, y = m.groups()
        y = y if len(y) == 4 else "20" + y
        return f"{y}-{int(mo):02d}-{int(d):02d}", "num", "date"
    if PCT_RE.match(w):
        return w, "num", "percent"
    if NUM_RE.match(w):
        return w, "num", "number"

    if w in PARTICLES:
        return w, "ms", "particle"
    if w in SLANG:
        return SLANG[w], "ms", "slang"

    # reduplication: budak2 / budak-budak
    if w.endswith("2") and len(w) > 3:
        return lemma_of(w[:-1])[0], "ms", "reduplication"
    if "-" in w:
        a, _, b = w.partition("-")
        if a == b and a:
            return lemma_of(a)[0], "ms", "reduplication"

    if w in ROOTS:
        return w, "ms", "dict"

    root, rule = strip_affix(w)
    if root:
        return root, "ms", "affix"          # NEEDS REVIEW

    col = normalise_repeat(w)
    if col:
        if col in ROOTS:
            return col, "ms", "elongation"
        if col in SLANG:
            return SLANG[col], "ms", "elongation"

    if w in EN_HINT or w in EN_VOCAB:
        return w, "en", "codeswitch"
    if not (set(w) & VOWELS) or re.search(r"[^a-z0-9'\-]", w):
        return w, "other", "unknown"        # NEEDS REVIEW

    if w in MS_VOCAB:
        # Known Malay, root not in ROOTS. Try affix stripping without the
        # root-lookup requirement - a proposal is more useful than a shrug.
        stripped, _ = strip_affix_loose(w)
        if stripped and stripped != w:
            return stripped, "ms", "affix_guess"     # NEEDS REVIEW
        return w, "ms", "ms_unlisted"                # NEEDS REVIEW

    return w, "en", "codeswitch_guess"      # NEEDS REVIEW


# ---------------------------------------------------------------------------
# CORPUS MINING
# ---------------------------------------------------------------------------

MARKERS = set(SLANG) | PARTICLES | {
    "yang", "dan", "ini", "itu", "saya", "kita", "dia", "di", "ke", "dari",
    "untuk", "dengan", "boleh", "ada", "kena", "lagi", "juga", "sangat",
    "memang", "bila", "kalau", "semua", "orang", "buat", "pergi", "balik",
    "beli", "jual", "banyak", "tidak", "bukan", "sudah", "mahu",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['\-.,/%][A-Za-z0-9]+)*|[A-Za-z0-9]+")


def tokenize(s):
    return TOKEN_RE.findall(s)


def split_sentences(t):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\.{2,}|\n|\t", t) if s.strip()]


def clean(s):
    s = unicodedata.normalize("NFKC", s)
    s = "".join(c for c in s if ord(c) < 0x2E80)      # drop CJK / emoji
    s = re.sub(r"https?://\S+", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" \"'“”·-")
    return s


def read_uploads():
    """Every raw line under sentidata/, as plain text."""
    raw = []
    for f in MIX:
        with open(f"{UPLOADS}/{f}", encoding="utf-8") as fh:
            raw += [l.strip() for l in fh if l.strip()]
    for f in TSV:
        with open(f"{UPLOADS}/{f}", encoding="utf-8") as fh:
            for l in fh:
                raw += split_sentences(l.rsplit("\t", 1)[0])
    return raw


def induce_from_uploads(raw=None, quiet=False):
    """Populate EN_VOCAB / MS_VOCAB from sentidata. Returns both.

    Pure-English lines vs Malay-dominant lines give evidence about which
    register each token belongs to. Split out of mine() so that inference can
    reach it: skill.derive_rule() has to answer "is this English?" for a token
    the model left unchanged, and the only defensible answer is the one the
    ANNOTATION used. Deciding it differently there would make derived tags
    disagree with the corpus labels for reasons that have nothing to do with
    the model. Costs about 0.06s and is done once per process.
    """
    raw = read_uploads() if raw is None else raw
    en_lines, ms_lines = [], []
    for s in raw:
        c = clean(s)
        toks = tokenize(c)
        if len(toks) < 3:
            continue
        hits = len(MARKERS & {t.lower() for t in toks})
        if hits == 0:
            en_lines.append(c)
        elif hits / len(toks) >= 0.25:
            ms_lines.append(c)
    induce_vocab(en_lines, ms_lines)
    if not quiet:
        print(f"induced vocab: {len(EN_VOCAB)} en / {len(MS_VOCAB)} ms "
              f"(from {len(en_lines)} en / {len(ms_lines)} ms lines)")
    return EN_VOCAB, MS_VOCAB


def mine():
    raw = read_uploads()
    induce_from_uploads(raw)

    seen, cands = set(), []
    for s in raw:
        s = clean(s)
        toks = tokenize(s)
        if not (3 <= len(toks) <= 20):
            continue
        low = {t.lower() for t in toks}
        score = len(MARKERS & low)
        if score < 1:
            continue
        key = " ".join(t.lower() for t in toks)
        if key in seen:
            continue
        seen.add(key)
        cands.append((score, len(toks), s, toks))

    # richest rojak first, shorter first as tiebreak
    cands.sort(key=lambda x: (-x[0], x[1]))
    return cands


# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "Anda adalah pelemma bahasa Melayu. Tukarkan ayat rojak kepada ayat "
    "berbentuk lema. Kembalikan ayat sahaja."
)


def main():
    # carry over the hand-checked rows if present
    existing = []
    try:
        with open("seed.jsonl", encoding="utf-8") as f:
            existing = [json.loads(l) for l in f if l.strip()]
    except FileNotFoundError:
        pass

    for r in existing:
        r.setdefault("lemma_text", " ".join(e["l"] for e in r["lemmas"]))
        r["origin"] = "handmade"

    need = TARGET_ROWS - len(existing)
    rows = list(existing)
    review = Counter()

    for i, (score, _, text, toks) in enumerate(mine()[:need], 1):
        lemmas, candidates = [], {}
        for t in toks:
            lem, lang, src = lemma_of(t)
            lemmas.append({"t": t, "l": lem, "lang": lang, "src": src})
            candidates[t] = [lem] if src in ("dict", "slang") else []
            if src in ("affix", "affix_guess", "unknown", "codeswitch_guess", "ms_unlisted"):
                review[(t.lower(), lem, src)] += 1
        rows.append({
            "id": f"auto_{i:04d}",
            "text": text,
            "tokens": toks,
            "candidates": candidates,
            "lemmas": lemmas,
            "lemma_text": " ".join(e["l"] for e in lemmas),
            "origin": "mined",
            "rojak_score": score,
        })

    with open("seed500.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open("train_chatml.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": r["text"]},
                {"role": "assistant", "content": r["lemma_text"]},
            ]}, ensure_ascii=False) + "\n")

    with open("review_queue.tsv", "w", encoding="utf-8") as f:
        f.write("count\ttoken\tproposed_lemma\trule\n")
        for (t, lem, src), n in review.most_common():
            f.write(f"{n}\t{t}\t{lem}\t{src}\n")

    counts = Counter(e["src"] for r in rows for e in r["lemmas"])
    total = sum(counts.values())
    unsure = sum(counts[k] for k in ("affix", "affix_guess", "unknown", "codeswitch_guess", "ms_unlisted"))

    print(f"{len(rows)} rows  ({len(existing)} handmade + {len(rows)-len(existing)} mined)")
    print(f"{total} tokens\n")
    for src, n in counts.most_common():
        flag = "  <-- REVIEW" if src in ("affix", "affix_guess", "unknown", "codeswitch_guess", "ms_unlisted") else ""
        print(f"  {src:<18} {n:>5}  ({n/total:>5.1%}){flag}")
    print(f"\n  tokens needing review: {unsure} ({unsure/total:.1%})")
    print(f"  unique tokens in review_queue.tsv: {len(review)}")
    print(f"\n  sample:\n    {rows[60]['text']}\n    -> {rows[60]['lemma_text']}")


if __name__ == "__main__":
    main()