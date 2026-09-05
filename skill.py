#!/usr/bin/env python
import argparse
import json
import os
import re
import sys

SKILL_VERSION = "v1-reasoning"

SKILL_FILE = "skill.json"


RULES = {
    "KAMUS":    "kata dasar, sudah berbentuk lema",
    "RINGKAS":  "singkatan atau slanga, dikembangkan",
    "IMBUHAN":  "imbuhan dibuang untuk dapat kata dasar",
    "GANDA":    "penggandaan diselesaikan kepada kata dasar",
    "PANJANG":  "huruf berulang dipendekkan",
    "PARTIKEL": "partikel wacana, sengaja tidak ditukar",
    "INGGERIS": "kata Inggeris, dikekalkan",
    "NOMBOR":   "nombor, masa, tarikh atau wang",
    "KEKAL":    "tidak dikenali, dikekalkan seadanya",
}

RULES_EN = {
    "KAMUS":    "already a root; unchanged",
    "RINGKAS":  "abbreviation or slang, expanded",
    "IMBUHAN":  "affix stripped to reach the root",
    "GANDA":    "reduplication resolved to its base",
    "PANJANG":  "elongated letters collapsed",
    "PARTIKEL": "discourse particle, deliberately untouched",
    "INGGERIS": "English token, kept as-is",
    "NOMBOR":   "number, time, date or money",
    "KEKAL":    "unrecognised, left alone",
}


SRC_TO_RULE = {
    "dict":       ("KAMUS",    False),
    "deriv":      ("KAMUS",    False),
    "slang":      ("RINGKAS",  False),
    "particle":   ("PARTIKEL", False),
    "num":        ("NOMBOR",   False),
    "redup":      ("GANDA",    False),
    "elong":      ("PANJANG",  False),
    "codeswitch": ("INGGERIS", False),
    "affix":      ("IMBUHAN",  True),

    "ambiguous":  ("KAMUS",    True),

    "dict_check": ("KEKAL",    True),
    "oov":        ("KEKAL",    True),
    "unknown":    ("KEKAL",    True),
}

HUMAN_SRC = "human"


SYSTEM_PROMPT = """\
Anda pelemma bahasa Melayu rojak. Untuk setiap token, beri lema dan sebabnya.

Format jawapan, tiada teks lain:
<lema>
token | PERATURAN | lema
</lema>
<ayat>ayat lema penuh</ayat>

PERATURAN:
KAMUS    kata dasar, kekal
RINGKAS  singkatan/slanga dikembangkan (dh->sudah, xleh->tidak boleh)
IMBUHAN  imbuhan dibuang (berjalan->jalan, memakan->makan)
GANDA    penggandaan jadi satu lema (budak2->budak)
PANJANG  huruf berulang dipendekkan (sedappp->sedap)
PARTIKEL partikel wacana, kekal (la, weh, kan, je)
INGGERIS kata Inggeris, kekal
NOMBOR   nombor, masa, tarikh, wang
KEKAL    tidak dikenali, kekal
Tulis "?" selepas peraturan jika tidak pasti, contoh: INGGERIS?

JANGAN:
- jangan terjemah ayat ke bahasa Inggeris
- jangan jawab soalan atau beri pendapat; lemakan sahaja
- satu baris bagi setiap token, ikut susunan asal, jangan tambah atau buang
- jangan tukar partikel atau nama khas
- jangan buang imbuhan daripada kata dasar (dengan, makan, kerana, sebab)
- jangan pecahkan penggandaan kepada dua perkataan
- jangan reka lema untuk kata yang tidak dikenali
- jangan tulis apa-apa di luar <lema> dan <ayat>"""


PROMPTS = {"v1-reasoning": SYSTEM_PROMPT}
DEFAULT_VERSION = SKILL_VERSION


def version_for(adapter_dir):

    meta = os.path.join(str(adapter_dir), SKILL_FILE)
    if os.path.exists(meta):
        try:
            with open(meta, encoding="utf-8") as fh:
                v = json.load(fh).get("skill_version")
            if v in PROMPTS:
                return v
        except (OSError, ValueError):
            pass          # an unreadable stamp is no worse than no stamp
    print("[skill] %s carries no %s -- assuming %s. Stamp it with "
          "write_version() to make this certain."
          % (adapter_dir, SKILL_FILE, DEFAULT_VERSION))
    return DEFAULT_VERSION


def prompt_for(adapter_dir):

    v = version_for(adapter_dir)
    return PROMPTS[v], v


def write_version(adapter_dir, version=SKILL_VERSION):
    if version not in PROMPTS:
        raise ValueError("unknown skill version %r" % (version,))
    path = os.path.join(str(adapter_dir), SKILL_FILE)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"skill_version": version, "skill_name": SKILL_VERSION},
                  fh, indent=2)
    return path


_REDUP_RE = re.compile(r"^(.+?)2$")
_ELONG_RE = re.compile(r"(.)\1{1,}")


def infer_rule(token, lemma):

    t, l = token.lower(), lemma.lower()
    if t == l:
        return ("KAMUS", False)
    m = _REDUP_RE.match(t)
    if m and m.group(1) == l:
        return ("GANDA", False)
    if "-" in t and t.split("-")[0] == l:
        return ("GANDA", False)
    if _ELONG_RE.sub(r"\1", t) == _ELONG_RE.sub(r"\1", l):
        return ("PANJANG", False)
    if t.endswith(l) or t.startswith(l):
        return ("IMBUHAN", False)          # a prefix or suffix came off
    return ("RINGKAS", True)               # a substitution, reason not visible


def rule_for(src, token=None, lemma=None):
    """(rule tag, uncertain) for one annotated token."""
    if src == HUMAN_SRC and token is not None and lemma is not None:
        return infer_rule(token, lemma)
    return SRC_TO_RULE.get(src, ("KEKAL", True))


def render_target(row):

    lines, words = [], []
    for e in row["lemmas"]:
        tag, uncertain = rule_for(e["src"], e["t"], e["l"])
        lines.append("%s | %s%s | %s"
                     % (e["t"], tag, "?" if uncertain else "", e["l"]))
        words.append(e["l"])
    return "<lema>\n%s\n</lema>\n<ayat>%s</ayat>" % ("\n".join(lines),
                                                    " ".join(words))


def to_chatml(row, system_prompt=SYSTEM_PROMPT):
    """A corpus row -> one ChatML training example."""
    return {"messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": row["text"]},
        {"role": "assistant", "content": render_target(row)},
    ]}


LEMA_BLOCK_RE = re.compile(r"<lema>(.*?)</lema>", re.S | re.I)
AYAT_BLOCK_RE = re.compile(r"<ayat>(.*?)</ayat>", re.S | re.I)
LINE_RE = re.compile(r"^\s*(\S+)\s*\|\s*([A-Za-z]+)(\??)\s*\|\s*(.+?)\s*$")


def parse_response(text):

    block = LEMA_BLOCK_RE.search(text)
    body = block.group(1) if block else text

    steps = []
    for line in body.splitlines():
        m = LINE_RE.match(line)
        if not m:
            continue
        token, rule, mark, lemma = m.groups()
        rule = rule.upper()
        steps.append({"token": token,
                      "rule": rule if rule in RULES else "KEKAL",
                      "uncertain": bool(mark) or rule not in RULES,
                      "lemma": lemma.strip()})

    ayat = AYAT_BLOCK_RE.search(text)
    if ayat:
        sentence = _collapse_ws(ayat.group(1))
    else:
        sentence = " ".join(s["lemma"] for s in steps)

    return {"steps": steps, "sentence": sentence,
            "ok": bool(block and steps)}


def _collapse_ws(s):
    return re.sub(r"\s+", " ", s).strip()


TAGS_EN = {
    "KAMUS":    "dictionary",
    "RINGKAS":  "slang expanded",
    "IMBUHAN":  "affix stripped",
    "GANDA":    "reduplication",
    "PANJANG":  "elongation",
    "PARTIKEL": "particle",
    "INGGERIS": "English",
    "NOMBOR":   "number",
    "KEKAL":    "unchanged",
}


def explain(rule, uncertain=False, lang="ms"):
    """One line of prose for a rule tag."""
    table = RULES if lang == "ms" else RULES_EN
    text = table.get(rule, table["KEKAL"])
    if uncertain:
        text += " (tidak pasti)" if lang == "ms" else " (uncertain)"
    return text


def format_result(parsed, lang="ms"):

    tags = TAGS_EN if lang == "en" else None
    out = [parsed["sentence"], ""]
    width = max((len(s["token"]) for s in parsed["steps"]), default=0)
    for s in parsed["steps"]:
        label = (tags or {}).get(s["rule"], s["rule"])
        out.append("  %-*s  ->  %-12s  %s%s"
                   % (width, s["token"], s["lemma"], label,
                      "?" if s["uncertain"] else ""))
    if not parsed["ok"]:
        out.append("")
        out.append("[format tidak lengkap]" if lang == "ms"
                   else "[malformed answer]")
    return "\n".join(out)


DEMO_RESPONSE = """\
<lema>
budak2 | GANDA | budak
tu | RINGKAS | itu
gilaaa | PANJANG | gila
</lema>
<ayat>budak itu gila</ayat>"""


def _load_rows(path, n):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= n:
                break
    return rows


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser()
    p.add_argument("--card", action="store_true", help="print the system prompt")
    p.add_argument("--rules", action="store_true", help="print the rule table")
    p.add_argument("--demo", action="store_true",
                   help="parse the built-in answer, or render --seed rows")
    p.add_argument("--parse", metavar="FILE",
                   help="parse a model answer from a file ('-' for stdin)")
    p.add_argument("--seed", metavar="JSONL",
                   help="corpus file to render training targets from")
    p.add_argument("-n", type=int, default=3, help="how many rows (default 3)")
    p.add_argument("--lang", default="ms", choices=("ms", "en"))
    a = p.parse_args()

    if a.card:
        print(SYSTEM_PROMPT)
        return
    if a.rules:
        print("%-9s %-42s %s" % ("TAG", "GLOSS (ms)", "GLOSS (en)"))
        print("-" * 92)
        for tag in RULES:
            print("%-9s %-42s %s" % (tag, RULES[tag], RULES_EN[tag]))
        print("\n%d src values map onto these tags:" % len(SRC_TO_RULE))
        for src, (tag, unc) in sorted(SRC_TO_RULE.items()):
            print("  %-12s -> %-9s%s" % (src, tag, "?" if unc else ""))
        return
    if a.parse:
        text = sys.stdin.read() if a.parse == "-" else \
            open(a.parse, encoding="utf-8").read()
        print(format_result(parse_response(text), a.lang))
        return
    if a.demo:
        if a.seed:
            for row in _load_rows(a.seed, a.n):
                print("INPUT: %s\n" % row["text"])
                print(render_target(row))
                print("\n" + "-" * 70 + "\n")
        else:
            print("MODEL ANSWER\n%s\n\nPARSED\n" % DEMO_RESPONSE)
            print(format_result(parse_response(DEMO_RESPONSE), a.lang))
        return
    p.print_help()


if __name__ == "__main__":
    main()
