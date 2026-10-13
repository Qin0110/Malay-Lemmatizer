#!/usr/bin/env python
"""Text cleaning shared by corpus building and the lemmatiser.

The model is trained on sentences that went through clean() and sentences(),
so the app and main.py send it text prepared exactly the same way.

    python preprocess.py "Budak2 tu gilaaa!! @ali https://t.co/x"
    python preprocess.py --demo
"""

import argparse
import html
import re
import sys
import unicodedata

URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
MENTION_RE = re.compile(r"@[\w.-]+")
IC_RE = re.compile(r"\b\d{6}-\d{2}-\d{4}\b")
PHONE_RE = re.compile(r"(?<!\w)(?:\+?6?0)1\d[\s-]?\d{3,4}[\s-]?\d{3,4}(?!\w)")
TIMESTAMP_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
ELLIPSIS_RE = re.compile(r"\.{2,}")
SENT_SPLIT_RE = re.compile(r"[.!?\n]+|\s{3,}")

# "budak2" has to be tried before the plain word, or it splits into "budak" + "2"
WORD_RE = re.compile(r"[A-Za-z]+\d|[A-Za-z]+(?:-[A-Za-z]+)*|\d+")


def clean(text):
    """Remove links, handles, emails, phone numbers, emoji and symbols."""
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = IC_RE.sub(" ", text)
    text = PHONE_RE.sub(" ", text)
    text = TIMESTAMP_RE.sub(" ", text)
    text = ELLIPSIS_RE.sub(" ", text)
    # symbols become a space rather than nothing, so "best\U0001f602aku" stays two words
    text = "".join(
        c if (c == "\n" or unicodedata.category(c)[0] not in ("S", "C"))
        else " "
        for c in text
    )
    text = re.sub(r"[^\S\n]+", " ", text)
    return "\n".join(ln.strip() for ln in text.split("\n")).strip()


def sentences(text):
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


def tokens(text):
    return WORD_RE.findall(text)


def prepare(text):
    """Raw input -> the list of sentences the lemmatiser works on."""
    return [s for s in sentences(clean(text)) if tokens(s)]


def trace(text):
    cleaned = clean(text)
    sents = [s for s in sentences(cleaned) if tokens(s)]
    return [("input", text), ("cleaned", cleaned), ("sentences", sents),
            ("tokens", [tokens(s) for s in sents])]


DEMO = [
    "Budak2 tu gilaaa!! 3 org je \U0001f602\U0001f602",
    "Zmn skrg org berahsia buat patent cik, dah tak main simpan mulut",
    "Sedappp gilerrr @chef_ali https://t.co/abc #foodie",
    "Safwan best la... Kak Uji pun ada kat podcast tu",
    "call aku 012-345 6789 atau email ali@contoh.com",
]


def show(text):
    for name, value in trace(text):
        print("  %-10s %s" % (name, value))
    print()


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(description="Show how input text is cleaned.")
    p.add_argument("sentence", nargs="*")
    p.add_argument("--demo", action="store_true", help="run the built-in examples")
    p.add_argument("--stdin", action="store_true", help="one sentence per line")
    a = p.parse_args()

    if a.demo:
        for s in DEMO:
            show(s)
    elif a.stdin:
        for line in sys.stdin:
            if line.strip():
                show(line.strip())
    elif a.sentence:
        show(" ".join(a.sentence))
    else:
        p.print_help()


if __name__ == "__main__":
    main()
