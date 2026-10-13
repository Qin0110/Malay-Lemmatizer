#!/usr/bin/env python
"""Import social media posts from export files into data/raw/.

    python crawl/import_posts.py tweets.js --source twitter
    python crawl/import_posts.py threads.json --source threads
    python crawl/import_posts.py comments.csv --source facebook --field message
    python crawl/import_posts.py posts.txt --source forum

Reads .txt (one post per line), .csv/.tsv, .json, .jsonl and the X/Twitter
archive's tweets.js. Only the post text is kept - no usernames, ids, dates or
links. @handles, emails and phone numbers are removed from the text before
it is saved.

Then: python crawl/build_corpus.py
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from preprocess import clean                                      # noqa: E402

RAW_DIR = ROOT / "data" / "raw"

TEXT_FIELDS = ("full_text", "text", "content", "body", "message", "caption",
               "comment", "tweet", "post", "title", "value")
RT_RE = re.compile(r"^RT\s+@\w+:\s*")


def strings_in(obj, fields, out):
    """Every string stored under one of `fields`, anywhere in a JSON tree."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key.lower() in fields and isinstance(value, str):
                out.append(value)
            else:
                strings_in(value, fields, out)
    elif isinstance(obj, list):
        for item in obj:
            strings_in(item, fields, out)
    return out


def read_json(path, fields):
    raw = path.read_text(encoding="utf-8-sig")
    if path.suffix == ".js":
        raw = raw[raw.index("["):]
    if path.suffix == ".jsonl":
        out = []
        for line in raw.splitlines():
            if line.strip():
                strings_in(json.loads(line), fields, out)
        return out
    return strings_in(json.loads(raw), fields, [])


def read_csv(path, field):
    delim = "\t" if path.suffix == ".tsv" else ","
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=delim)
        cols = {c.lower(): c for c in reader.fieldnames or []}
        names = [field] if field else TEXT_FIELDS
        col = next((cols[n.lower()] for n in names if n.lower() in cols), None)
        if col is None:
            sys.exit("%s: no text column found (columns: %s). Use --field."
                     % (path.name, ", ".join(reader.fieldnames or [])))
        return [row[col] for row in reader if row.get(col)]


def read_posts(path, field=None):
    if path.suffix in (".txt",):
        return [l for l in path.read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    if path.suffix in (".csv", ".tsv"):
        return read_csv(path, field)
    if path.suffix in (".json", ".jsonl", ".js"):
        fields = {field.lower()} if field else set(TEXT_FIELDS)
        return read_json(path, fields)
    sys.exit("%s: unsupported file type (use .txt, .csv, .tsv, .json, .jsonl or .js)"
             % path.name)


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(description="Import posts into data/raw/.")
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("--source", required=True,
                   help="where the posts came from: twitter, threads, facebook, "
                        "instagram, reddit, tiktok, forum, ...")
    p.add_argument("--field", help="name of the text column / JSON key")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    source = a.source.strip().lower()
    out_path = RAW_DIR / ("%s.jsonl" % re.sub(r"[^a-z0-9_-]", "", source))
    have = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as fh:
            have = {json.loads(l)["text"] for l in fh if l.strip()}

    new = []
    for path in a.files:
        if not path.exists():
            sys.exit("no such file: %s" % path)
        posts = read_posts(path, a.field)
        kept = 0
        for text in posts:
            text = clean(RT_RE.sub("", text))
            if text and text not in have:
                have.add(text)
                new.append(text)
                kept += 1
        print("%-30s %6d posts, %6d new" % (path.name, len(posts), kept))

    if a.dry_run:
        for text in new[:5]:
            print("  | " + " ".join(text.split())[:100])
        print("\n--dry-run: nothing written.")
        return
    if not new:
        print("nothing new to add")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as fh:
        for text in new:
            fh.write(json.dumps({"text": text, "source": source},
                                ensure_ascii=False) + "\n")
    print("\nadded %d posts -> %s" % (len(new), out_path.relative_to(ROOT)))
    print("Next: python crawl/build_corpus.py")


if __name__ == "__main__":
    main()
