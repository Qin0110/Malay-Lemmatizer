import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "crawl"))
sys.path.insert(0, str(ROOT / "unsloth"))

DEFAULT_FILE = ROOT / "data" / "gold.jsonl"
FROM_SRC = "oov"
TO_SRC = "codeswitch"


def main(path, dry_run):
    from make_goldsheet import looks_english

    if not path.exists():
        sys.exit("No %s" % path)

    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]

    before = Counter()
    changed, skipped, examples = 0, 0, []
    for row in rows:
        for e in row["lemmas"]:
            before[e["src"]] += 1
            if e["src"] != FROM_SRC or not looks_english(e["t"]):
                continue
            if e["l"].lower() != e["t"].lower():
                skipped += 1
                continue
            e["src"] = TO_SRC
            changed += 1
            if len(examples) < 15:
                examples.append(e["t"])

    after = Counter(e["src"] for r in rows for e in r["lemmas"])
    n_tok = sum(before.values())

    print("%s: %d sentences, %d tokens\n" % (path.name, len(rows), n_tok))
    print("%-12s %8s %8s" % ("src", "before", "after"))
    print("-" * 30)
    for src in sorted(set(before) | set(after)):
        mark = "  <-" if before[src] != after[src] else ""
        print("%-12s %8d %8d%s" % (src, before[src], after[src], mark))

    print("\nretagged %s -> %s : %d tokens (%.1f%%)"
          % (FROM_SRC, TO_SRC, changed, 100 * changed / max(n_tok, 1)))
    if skipped:
        print("skipped %d English `oov` tokens whose lemma differs from the "
              "token - left for a human." % skipped)
    print("examples:", ", ".join(examples))

    # The invariant this whole script rests on.
    lemmas_ok = all(e["l"] for r in rows for e in r["lemmas"])
    rebuilt_ok = all(
        " ".join(e["l"] for e in r["lemmas"]) == r["lemma_text"]
        for r in rows if "lemma_text" in r)
    print("\nlemmas intact                 %s" % ("PASS" if lemmas_ok else "FAIL"))
    print("lemma_text still reconstructs %s"
          % ("PASS" if rebuilt_ok else "FAIL"))
    if not (lemmas_ok and rebuilt_ok):
        sys.exit("\nInvariant broken - nothing written.")

    if dry_run:
        print("\n--dry-run: nothing written.")
        return
    if not changed:
        print("\nNothing to change.")
        return

    backup = path.with_suffix(path.suffix + ".pre_retag")
    if not backup.exists():
        shutil.copy2(path, backup)
        print("\nbacked up -> %s" % backup.name)

    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("wrote %s" % path.relative_to(ROOT))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--file", default=str(DEFAULT_FILE),
                   help="labelled split to retag (default data/gold.jsonl)")
    p.add_argument("--dry-run", action="store_true",
                   help="report the change without writing it")
    a = p.parse_args()
    main(Path(a.file), a.dry_run)
