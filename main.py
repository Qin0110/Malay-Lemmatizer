#!/usr/bin/env python
"""Malay rojak lemmatiser in the terminal.

    python main.py "aku dh xleh handle keje tu"      one sentence
    python main.py                                    interactive
    python main.py --rules-only "Budak2 tu gilaaa"    no GPU needed
    python main.py --model qwen-ft "aq nk tdr"
"""

import argparse
import sys

import pipeline
import skill
from preprocess import trace

BANNER = """\
Type a sentence, or one of:
  /rules    toggle following the training rules     /stages   toggle the cleaning trace
  /plain    toggle sentence-only output              /lang     toggle ms/en glosses
  quit      exit
"""


def show(results, plain=False, lang="ms"):
    for res in results:
        if plain:
            print(res["sentence"])
            continue
        print("\n  " + res["sentence"] + "\n")
        width = max(len(s["token"]) for s in res["steps"])
        for s in res["steps"]:
            rule = s["rule"] + ("?" if s["uncertain"] else "")
            gloss = skill.explain(s["rule"], lang=lang)
            extra = ""
            if s.get("model_lemma"):
                extra = "   (model said %s)" % s["model_lemma"]
            elif s.get("note"):
                extra = "   (%s)" % s["note"]
            print("  %-*s -> %-16s %-9s %-6s %s%s"
                  % (width, s["token"], s["lemma"], rule, s["by"], gloss, extra))
        if res["ok"] is False:
            print("\n  [the model's answer was malformed - rule labels used where needed]")
    print()


def show_stages(text):
    for name, value in trace(text):
        print("  %-10s %s" % (name, value))


def pick_model(name, rules_only):
    if rules_only:
        return None
    if name is None:
        on_disk = pipeline.adapters_on_disk()
        if not on_disk:
            print("[no fine-tuned adapter in models/ - using the rules only]\n")
            return None
        name = on_disk[0]
    if not pipeline.gpu_packages():
        print("[GPU packages are not installed - using the rules only. "
              "Run SETUP.bat gpu to add them.]\n")
        return None
    print("Loading %s..." % name)
    model = pipeline.Model(name)
    print("Ready (prompt %s).\n" % model.version)
    return model


def repl(model, follow, plain, lang, stages):
    print(BANNER)
    while True:
        try:
            text = input("rojak> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not text:
            continue
        if text.lower() in ("quit", "exit", "q"):
            return
        if text == "/rules":
            follow = not follow
            print("follow the training rules: %s\n" % ("ON" if follow else "OFF"))
            continue
        if text == "/stages":
            stages = not stages
            continue
        if text == "/plain":
            plain = not plain
            continue
        if text == "/lang":
            lang = "en" if lang == "ms" else "ms"
            continue
        if stages:
            show_stages(text)
        try:
            show(pipeline.lemmatise(text, model, follow), plain, lang)
        except KeyboardInterrupt:
            print("\n  (interrupted)\n")


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(
        description="Lemmatise informal Malay.",
        epilog="models: %s" % ", ".join(list(pipeline.ADAPTERS) + list(pipeline.BASE_MODELS)))
    p.add_argument("sentence", nargs="*")
    p.add_argument("--model", help="which model (default: the first adapter in models/)")
    p.add_argument("--rules-only", action="store_true", help="skip the model")
    p.add_argument("--no-follow-rules", action="store_true",
                   help="show the model's own answer for every token")
    p.add_argument("--stages", action="store_true", help="show how the input is cleaned")
    p.add_argument("--plain", action="store_true", help="print the lemma sentence only")
    p.add_argument("--lang", default="ms", choices=("ms", "en"))
    a = p.parse_args()

    model = pick_model(a.model, a.rules_only)
    follow = not a.no_follow_rules

    if a.sentence:
        text = " ".join(a.sentence)
        if a.stages:
            show_stages(text)
        results = pipeline.lemmatise(text, model, follow)
        if not results:
            sys.exit("nothing left to lemmatise after cleaning")
        show(results, a.plain, a.lang)
    else:
        repl(model, follow, a.plain, a.lang, a.stages)


if __name__ == "__main__":
    main()
