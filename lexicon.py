import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
USER_FILE = ROOT / "data" / "user_lexicon.csv"
SHIPPED_FILE = ROOT / "malindo_morph" / "BM_dict.csv"
NAMES_FILE = ROOT / "data" / "user_names.txt"
HEADER = ("rojak", "actual")

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
    "lg": "lagi", "kak": "kakak", "tpi": "tetapi", "kiteorg": "kami",
    "jd": "jadi", "betol": "betul", "tngok": "tengok", "tgn": "tangan",
    "xjadi": "tidak jadi", "xtau": "tidak tahu", "xbleh": "tidak boleh",
    "xley": "tidak boleh", "xske": "tidak suka", "xyh": "tidak payah",
}

PARTICLES = {
    "la", "lah", "le", "lor", "leh", "meh", "weh", "wei", "woi",
    "je", "jer", "kan", "kot", "eh", "ah", "ha", "hah", "ke", "kah",
    "pun", "tau", "doh", "bah", "yo", "sia",
    "beb", "siak", "wey", "weyh", "bro", "bre",
}

# words that usually sit right in front of a name: Kak Uji, Pak Nil, Dato Rosyam
TITLES = {
    "kak", "akak", "abang", "abg", "bang", "adik", "dik", "pak", "pakcik",
    "makcik", "mak", "cik", "encik", "puan", "tuan", "dato", "datuk", "datin",
    "tan", "sri", "tok", "haji", "hajah", "ustaz", "ustazah", "dr", "prof",
    "wak", "tun", "tunku", "tengku", "raja", "sultan", "allahyarham",
    "allahyarhamah", "almarhum", "almarhumah", "mendiang",
    "uncle", "auntie", "aunty", "mr", "mrs", "ms", "sis",
}

_CACHE = None
_NAMES = None


def _read(path):
    out = {}
    if not path.exists():
        return out
    try:
        with path.open(encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                key = (row.get("rojak") or "").strip().lower()
                val = (row.get("actual") or "").strip()
                if key and val and key not in out:
                    out[key] = val
    except (OSError, csv.Error) as exc:
        print("[lexicon] could not read %s: %s" % (path.name, exc))
    return out


def slang_map(reload=False):
    """rojak -> standard form. Your file beats SLANG, which beats BM_dict.csv."""
    global _CACHE
    if _CACHE is None or reload:
        merged = _read(SHIPPED_FILE)
        merged.update(SLANG)
        merged.update(_read(USER_FILE))
        _CACHE = merged
    return _CACHE


def names(reload=False):
    """Your proper nouns, one per line in data/user_names.txt, as word tuples."""
    global _NAMES
    if _NAMES is None or reload:
        from preprocess import WORD_RE
        out = set()
        if NAMES_FILE.exists():
            for line in NAMES_FILE.read_text(encoding="utf-8-sig").splitlines():
                words = tuple(w.lower() for w in WORD_RE.findall(line))
                if words and not line.lstrip().startswith("#"):
                    out.add(words)
        _NAMES = sorted(out, key=len, reverse=True)
    return _NAMES


def add_name(name):
    name = " ".join(name.split())
    if not name:
        sys.exit("give the name to add")
    from preprocess import WORD_RE
    if tuple(w.lower() for w in WORD_RE.findall(name)) in names():
        print("already listed: %s" % name)
        return
    NAMES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with NAMES_FILE.open("a", encoding="utf-8") as fh:
        fh.write(name + "\n")
    names(reload=True)
    print("added name: %s   (%s)" % (name, NAMES_FILE.name))


def lookup(token):
    return slang_map().get(token.strip().lower())


def source_of(token):
    key = token.strip().lower()
    if key in _read(USER_FILE):
        return "yours"
    if key in SLANG:
        return "built-in"
    if key in _read(SHIPPED_FILE):
        return "BM_dict"
    return None


def add(token, lemma):
    token, lemma = token.strip().lower(), lemma.strip()
    if not token or not lemma:
        sys.exit("both a token and a lemma are required")

    existing = _read(USER_FILE)
    if existing.get(token) == lemma:
        print("already present: %s -> %s" % (token, lemma))
        return
    if token in existing:
        print("%s is already mapped to %r in %s - edit that line instead."
              % (token, existing[token], USER_FILE.name))
        return

    USER_FILE.parent.mkdir(parents=True, exist_ok=True)
    new = not USER_FILE.exists() or USER_FILE.stat().st_size == 0
    with USER_FILE.open("a", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(HEADER)
        w.writerow([token, lemma])
    slang_map(reload=True)
    print("added: %s -> %s   (%s)" % (token, lemma, USER_FILE.name))
    print("Rebuild the training data so the model learns it too:")
    print("    python crawl/build_silver.py")
    print("    python crawl/build_sft.py")


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(description="Look up or extend the slang lexicon.")
    p.add_argument("--check", nargs="+", metavar="TOKEN",
                   help="show what each token expands to, and from where")
    p.add_argument("--add", nargs=2, metavar=("TOKEN", "LEMMA"),
                   help="add a mapping to data/user_lexicon.csv")
    p.add_argument("--add-name", metavar="NAME",
                   help='always keep this as a proper noun, e.g. "Berjaya Grocer"')
    p.add_argument("--list", action="store_true",
                   help="list your own entries and the totals")
    a = p.parse_args()

    if a.add:
        add(*a.add)
    elif a.add_name:
        add_name(a.add_name)
    elif a.check:
        for t in a.check:
            want = lookup(t)
            if want:
                print("  %-12s -> %-16s (%s)" % (t, want, source_of(t)))
            else:
                print("  %-12s -> (not in the lexicon)" % t)
    elif a.list:
        user, shipped = _read(USER_FILE), _read(SHIPPED_FILE)
        print("%-18s %4d entries" % (SHIPPED_FILE.name, len(shipped)))
        print("%-18s %4d entries" % ("built-in SLANG", len(SLANG)))
        print("%-18s %4d entries" % (USER_FILE.name, len(user)))
        for k in sorted(user):
            print("  %-14s -> %s" % (k, user[k]))
        print("\nmerged total: %d" % len(slang_map(reload=True)))
        print("%-18s %4d names" % (NAMES_FILE.name, len(names())))
    else:
        p.print_help()


if __name__ == "__main__":
    main()
