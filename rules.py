#!/usr/bin/env python
"""The rule-based lemmatiser.

It writes the labels the model is trained on, and the app uses it to keep the
model's answer in line with those labels.

    python rules.py "Aku dh xleh handle keje tu, Kak Uji"
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import lexicon                                                    # noqa: E402
from preprocess import WORD_RE                                    # noqa: E402

MALINDO_TSV = ROOT / "malindo_morph" / "malindo_dic_2024.tsv"
CORPUS = ROOT / "data" / "corpus.jsonl"

ELONG_RE = re.compile(r"([a-z])\1{2,}")
REDUP_RE = re.compile(r"^([a-z]{3,})2$")
VOWELS = set("aeiou")

# src values whose label the rules are sure about; the rest are a best guess
CERTAIN = {"num", "particle", "slang", "redup", "elong", "dict", "deriv",
           "codeswitch", "name"}

_MALINDO = None
_EN = None


def malindo():
    """(lemmas, roots, proper) from MALINDO Morph, loaded once.

    lemmas  surface -> list of lemma cells
    roots   surface -> root
    proper  words MALINDO only has as a capitalised headword (Malaysia, Allah)
    """
    global _MALINDO
    if _MALINDO is None:
        import pandas as pd
        df = pd.read_csv(MALINDO_TSV, sep="\t", header=None, usecols=[1, 2, 9],
                         names=["root", "surface", "lemma"], encoding="utf-8",
                         dtype=str)
        heads = df[["root", "surface"]].dropna()
        heads = heads[heads["root"] == heads["surface"]]["root"]
        proper = ({h.lower() for h in heads if h[:1].isupper()}
                  - {h for h in heads if h == h.lower()})

        lemmas = defaultdict(list)
        sub = df[["surface", "lemma"]].dropna()
        for surface, lemma in zip(sub["surface"].str.lower(), sub["lemma"]):
            if lemma not in lemmas[surface]:
                lemmas[surface].append(lemma)

        roots = {}
        sub = df[["root", "surface"]].dropna()
        for surface, root in zip(sub["surface"].str.lower(), sub["root"].str.lower()):
            roots.setdefault(surface, root)

        _MALINDO = (lemmas, roots, proper | capitalised_in_corpus())
    return _MALINDO


def capitalised_in_corpus(min_count=3, share=0.7):
    """Words the corpus writes with a capital at least 70% of the time mid-sentence."""
    cap, low = defaultdict(int), defaultdict(int)
    if CORPUS.exists():
        with CORPUS.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                for word in WORD_RE.findall(json.loads(line)["text"])[1:]:
                    if word[:1].isupper() and not word.isupper():
                        cap[word.lower()] += 1
                    elif word.islower():
                        low[word] += 1
    return {w for w, n in cap.items()
            if n >= min_count and n >= share * (n + low[w])}


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


def headword(token, lex):
    if token not in lex:
        return False
    lemma, _ = pick(lex[token], token)
    return lemma == token


def is_derived_noun(token, root, lex):
    if suffix_of(token, root) in ("kan", "i"):
        return False
    if not headword(token, lex):
        return False
    if suffix_of(token, root) == "an":
        return True
    return token.startswith(("pe", "ke"))


def looks_nonstandard(token):
    letters = [c for c in token if c.isalpha()]
    if len(letters) < 2:
        return True
    return sum(1 for c in letters if c in VOWELS) / len(letters) < 0.35


def propose(token, lex=None, roots=None):
    """(lemma, src) for one lowercase token."""
    if lex is None:
        lex, roots, _ = malindo()
    if token.isdigit():
        return token, "num"
    if token in lexicon.PARTICLES:
        return token, "particle"
    want = lexicon.lookup(token)
    if want:
        return want, "slang"

    m = REDUP_RE.match(token)
    if m:
        base = m.group(1)
        lemma, _ = pick(lex.get(base, []), base)
        return (lemma or base), "redup"

    collapsed = ELONG_RE.sub(r"\1", token)
    if collapsed != token:
        want = lexicon.lookup(collapsed)
        if want:
            return want, "elong"
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


def is_name(word, prev="", first=False, lex=None, proper=None):
    """Is this token a proper noun (kata nama khas)?

    Case is the main evidence, so it has to be judged on the text as written,
    before anything is lowercased.
    """
    if lex is None:
        lex, _, proper = malindo()
    low = word.lower()
    if word == low:
        return False
    if len(low) < 2 or not low.replace("-", "").isalpha():
        return False
    if (low in lexicon.PARTICLES or low in lexicon.TITLES
            or lexicon.lookup(low) or ELONG_RE.search(low)):
        return False
    # long ALL-CAPS words are usually shouting, not acronyms
    if word.isupper() and len(word) > 5 and low not in proper:
        return False
    if low in proper:
        return True
    if prev and prev.lower() in lexicon.TITLES:
        return True
    if first:
        return False
    if looks_english(low) or (len(low) == 2 and english_dict()
                              and english_dict().check(low)):
        return False
    if low in lex:
        return False
    return True


# Malay function words end a name run; "Uji Dan Safwan" is two names
FUNCTION_WORDS = {"dan", "atau", "yang", "di", "ke", "dari", "daripada", "dengan",
                  "untuk", "pada", "kepada", "oleh", "bagi", "dalam", "ini", "itu",
                  "tapi", "tetapi"}


def in_name_run(word, lex):
    """Can this word be part of a multi-word name like Berjaya Grocer?"""
    low = word.lower()
    if not word[:1].isupper() or not low.replace("-", "").isalpha():
        return False
    if low in FUNCTION_WORDS or low in lexicon.PARTICLES or low in lexicon.TITLES \
            or lexicon.lookup(low) or ELONG_RE.search(low):
        return False
    # all caps: short acronyms (KL, JDT) yes, shouted words (BEST, GILA) no
    if word.isupper() and len(word) > 3:
        return len(word) <= 5 and low not in lex and not looks_english(low)
    return True


def trim_run(run, words, lex, roots, proper):
    """Drop ordinary words stuck to the edge of a real name ("Semoga Allah")."""
    def changes(i):
        low = words[i].lower()
        return propose(low, lex, roots)[0] != low

    def standalone(i):
        return is_name(words[i], words[i - 1] if i else "", i == 0, lex, proper)

    while len(run) >= 2 and changes(run[0]) and standalone(run[1]):
        run = run[1:]
    while len(run) >= 2 and changes(run[-1]) and standalone(run[-2]):
        run = run[:-1]
    # a run of nothing but ordinary Malay words ("Semoga Tuhan Merahmati") is
    # more likely a capitalised phrase than a name, so its affixed words keep
    # their normal lemma
    if all(words[i].lower() in lex and not looks_english(words[i].lower()) for i in run):
        run = [i for i in run if not changes(i)]
    return run


def name_spans(words, lex, roots, proper):
    """Positions of words inside multi-word names.

    Two or more capitalised words in a row count as one name (Berjaya Grocer,
    Kuala Lumpur, Mydin Mall), and so does anything in data/user_names.txt
    whatever its case. Capitals are ignored when nearly every word has one,
    because then they are just how the sentence was typed.
    """
    lowered = tuple(w.lower() for w in words)
    out = set()
    for phrase in lexicon.names():
        n = len(phrase)
        for i in range(len(words) - n + 1):
            if lowered[i:i + n] == phrase:
                out.update(range(i, i + n))

    rest = [w for w in words[1:] if w.isalpha()]
    if len(rest) >= 4 and sum(1 for w in rest if w[:1].isupper()) >= 0.8 * len(rest):
        return out

    run = []
    for i, word in enumerate(words + [""]):
        if word and in_name_run(word, lex):
            run.append(i)
            continue
        # every sentence starts with a capital, so an ordinary first word
        # ("Bergegar EMI ...") is not evidence of a name
        first = words[0].lower()
        if run and run[0] == 0 and (first in lex or looks_english(first)):
            run = run[1:]
        run = trim_run(run, words, lex, roots, proper)
        if len(run) >= 2:
            out.update(run)
        run = []
    return out


# "pgi" and "pg" are both pagi (morning) and pergi (go); the neighbours decide
GO_BEFORE = {"nak", "nk", "mahu", "mau", "nok", "jom", "tak", "tk", "x", "boleh",
             "blh", "kena", "akan", "nanti", "nnti", "mesti", "msti", "jangan",
             "jgn", "aku", "aq", "sy", "saya", "kita", "kami", "dia", "korang"}
GO_AFTER = {"ke", "kat", "kt", "dekat", "dkt", "di", "sana", "situ", "mana",
            "balik", "blk"}
TIME_BEFORE = {"esok", "esk", "semalam", "smlm", "td", "tadi", "pukul", "pkl",
               "jam", "setiap", "tiap", "awal", "selamat", "hari", "dri", "dari"}


def pagi_or_pergi(prev, nxt, next_is_name):
    if prev.isdigit() or prev in TIME_BEFORE:
        return "pagi"
    if prev in GO_BEFORE or nxt in GO_AFTER or next_is_name:
        return "pergi"
    return None


def label(text):
    """Label every token of a sentence: [{"t": token, "l": lemma, "src": rule}]."""
    lex, roots, proper = malindo()
    words = WORD_RE.findall(text)
    spans = name_spans(words, lex, roots, proper)
    names = [i in spans or is_name(w, words[i - 1] if i else "", i == 0, lex, proper)
             for i, w in enumerate(words)]

    out = []
    for i, word in enumerate(words):
        low = word.lower()
        if names[i]:
            out.append({"t": word, "l": word, "src": "name"})
            continue
        if low in ("pgi", "pg"):
            prev = words[i - 1].lower() if i else ""
            nxt = words[i + 1].lower() if i + 1 < len(words) else ""
            sense = pagi_or_pergi(prev, nxt, i + 1 < len(words) and names[i + 1])
            if sense:
                out.append({"t": word, "l": sense, "src": "slang"})
                continue
        lemma, src = propose(low, lex, roots)
        out.append({"t": word, "l": lemma, "src": src})
    return out


def lemma_text(labels):
    return " ".join(e["l"] for e in labels)


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    text = " ".join(sys.argv[1:]) or "Aku dh xleh handle keje tu, Kak Uji"
    labels = label(text)
    for e in labels:
        print("  %-14s %-16s %s" % (e["t"], e["l"], e["src"]))
    print("\n  " + lemma_text(labels))
