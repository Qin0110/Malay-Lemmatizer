#!/usr/bin/env python
"""
One worked example from every corpus the pipeline reads.

    cd unsloth
    python sample_sources.py                    # screenshot-friendly, 78 cols
    python sample_sources.py --markdown         # paste straight into the report
    python sample_sources.py --source gold -n 2 # just one corpus, two rows
    python sample_sources.py --raw-only         # only the mined sentiment files

WHY A SCRIPT AND NOT SCREENSHOTS
--------------------------------
The Methodology chapter has to show what each source actually looks like. Doing
that by hand means opening six files in an editor, and an editor screenshot
proves only that a file was opened -- it cannot show what the REGISTRY makes of
the file, which is the part that matters. This reads each corpus through
sources.load(), the same call llm.py --prepare uses, so the "on disk" and
"after conversion" halves are guaranteed to be the same pair the training data
was built from.

It also stays correct when a source changes. A screenshot does not.

Output is 78 columns so it fits a portrait page at 8pt without wrapping.
"""

import argparse
import json
import os
import sys
import textwrap

import databuilder
import skill
import sources

WIDTH = 78


# ---------------------------------------------------------------------------
# LAYOUT
# ---------------------------------------------------------------------------

def rule(char="-", label=""):
    if not label:
        return char * WIDTH
    head = "%s %s " % (char * 2, label)
    return head + char * max(0, WIDTH - len(head))


def banner(label):
    return "\n%s\n  %s\n%s" % ("=" * WIDTH, label, "=" * WIDTH)


def block(text, indent="  ", limit=None):
    """Indent a chunk, hard-wrapping anything past the page width."""
    out = []
    lines = text.rstrip("\n").split("\n")
    if limit is not None and len(lines) > limit:
        lines = lines[:limit] + ["... (%d more lines)" % (len(lines) - limit)]
    for line in lines:
        if len(line) + len(indent) <= WIDTH:
            out.append(indent + line)
        else:
            out += textwrap.wrap(line, WIDTH - len(indent) - 2,
                                 initial_indent=indent,
                                 subsequent_indent=indent + "  ") or [indent]
    return "\n".join(out)


def token_table(lemmas, limit=14):
    """t -> l with the rule tag, the way skill.py renders a derivation."""
    rows = lemmas[:limit]
    w_t = max([len(e["t"]) for e in rows] + [5])
    w_l = max([len(e["l"]) for e in rows] + [5])
    out = []
    for e in rows:
        if e.get("rule"):
            tag, unsure = e["rule"], bool(e.get("unsure"))
        else:
            tag, unsure = skill.rule_for(e.get("src"), e["t"], e["l"])
        out.append("  %-*s -> %-*s  %-9s  (src=%s)"
                   % (w_t, e["t"], w_l, e["l"], tag + ("?" if unsure else ""),
                      e.get("src") or e.get("rule") or "-"))
    if len(lemmas) > limit:
        out.append("  ... (%d more tokens)" % (len(lemmas) - limit))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# RAW MINING POOL  (sources 1 and 2 in the report's Table 1)
# ---------------------------------------------------------------------------

def show_raw(n, emit):
    emit(banner("RAW MINING POOL  -  databuilder.read_uploads()"))
    emit("  location: %s" % databuilder.UPLOADS)

    if not os.path.isdir(databuilder.UPLOADS):
        emit("  MISSING -- run this from the unsloth/ directory, or check that")
        emit("  sentidata/ sits next to databuilder.py.")
        return

    for fname in databuilder.MIX:
        path = os.path.join(databuilder.UPLOADS, fname)
        emit(rule("-", "%s  (plain text, one review per line)" % fname))
        if not os.path.exists(path):
            emit("  MISSING: %s" % path)
            continue
        with open(path, encoding="utf-8") as fh:
            lines = [l.strip() for l in fh if l.strip()][:n]
        for line in lines:
            emit(block(line, "  | "))

    for fname in databuilder.TSV:
        path = os.path.join(databuilder.UPLOADS, fname)
        emit(rule("-", "%s  (TSV: text <TAB> sentiment label)" % fname))
        if not os.path.exists(path):
            emit("  MISSING: %s" % path)
            continue
        with open(path, encoding="utf-8") as fh:
            lines = [l.rstrip("\n") for l in fh if l.strip()][:n]
        for line in lines:
            text, _, label = line.rpartition("\t")
            emit(block("text : %s" % text, "  | "))
            emit(block("label: %s   <- discarded; only the text column is mined"
                       % label, "  | "))

    # What the selection filter does to that raw text.
    emit(rule("-", "after mine(): cleaned, filtered, scored"))
    try:
        cands = sources_mine_preview(n)
    except Exception as exc:                     # missing files, bad encoding
        emit("  could not run mine(): %s" % exc)
        return
    for score, ntok, text in cands:
        emit(block("rojak_score=%d  tokens=%d" % (score, ntok), "  | "))
        emit(block(text, "  |   "))


def sources_mine_preview(n):
    """Top-n mined candidates, without writing anything."""
    out = []
    for score, ntok, text, _toks in databuilder.mine()[:n]:
        out.append((score, ntok, text))
    return out


# ---------------------------------------------------------------------------
# REGISTRY SOURCES  (sources 3-6 in the report's Table 1)
# ---------------------------------------------------------------------------

def raw_record(src, index):
    """The record as it sits on disk, before any conversion."""
    with open(src.path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if line.strip() and i >= index:
                return json.loads(line)
    return None


def show_source(src, n, emit):
    emit(banner("SOURCE '%s'  -  kind=%s%s"
                % (src.name, src.kind, "" if src.enabled else "  [DISABLED]")))
    emit("  path : %s" % src.path)
    emit("  note : %s" % src.note)

    if not src.exists():
        emit("  MISSING -- nothing to show. sources.py declares it, so the "
             "registry")
        emit("  reports it as MISSING rather than silently dropping it.")
        return
    if not src.enabled:
        emit("  Declared but not linked, so none of these rows reach training.")

    rows = []
    for row in sources.load(src):
        rows.append(row)
        if len(rows) >= n:
            break
    if not rows:
        emit("  file exists but yielded no rows")
        return

    for k, row in enumerate(rows):
        if n > 1:
            emit(rule("-", "row %d of %d" % (k + 1, n)))

        emit(rule(".", "1. ON DISK"))
        raw = raw_record(src, k)
        if raw is None:
            emit("  (could not re-read the raw line)")
        elif src.kind == "prose":
            instr, sent, answer = sources._triple(raw)
            emit("  keys      : %s" % ", ".join(sorted(raw)))
            emit(block("instruction: %s" % instr, "  ", limit=4))
            emit(block("input      : %s" % sent, "  "))
            emit(block("output     : %s" % answer, "  ", limit=10))
        else:
            emit("  keys      : %s" % ", ".join(sorted(raw)))
            emit(block("text      : %s" % raw.get("text", ""), "  "))
            emit(block("lemmas[0] : %s"
                       % json.dumps(raw["lemmas"][0], ensure_ascii=False), "  "))

        emit(rule(".", "2. AFTER sources.load()  -  per-token, rule-tagged"))
        emit("  text: %s" % row["text"])
        emit(token_table(row["lemmas"]))

        emit(rule(".", "3. TRAINING TARGET  -  skill.render_target()"))
        emit(block(skill.render_target(row), "  ", limit=18))


# ---------------------------------------------------------------------------

def markdown(text):
    """Wrap the whole report in a fenced block so it pastes into the write-up."""
    return "```text\n%s\n```" % text.strip("\n")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--source", help="only this source (see sources.py)")
    p.add_argument("-n", type=int, default=1, help="rows per source")
    p.add_argument("--raw-only", action="store_true",
                   help="only the sentiment files that feed mine()")
    p.add_argument("--no-raw", action="store_true",
                   help="skip the mining pool, show the registry only")
    p.add_argument("--markdown", action="store_true",
                   help="print inside a fenced code block")
    a = p.parse_args()

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    buf = []
    emit = buf.append

    if not a.no_raw and not a.source:
        show_raw(a.n, emit)
    if not a.raw_only:
        picked = [s for s in sources.SOURCES
                  if a.source is None or s.name == a.source]
        if a.source and not picked:
            raise SystemExit("unknown source %r; try: %s"
                             % (a.source, ", ".join(s.name
                                                    for s in sources.SOURCES)))
        for src in picked:
            show_source(src, a.n, emit)

    text = "\n".join(buf)
    print(markdown(text) if a.markdown else text)


if __name__ == "__main__":
    main()