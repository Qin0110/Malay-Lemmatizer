#!/usr/bin/env python
"""
Chat with the Malay rojak lemmatiser.

    python main.py                                  # interactive
    python main.py "aku dh xleh handle keje tu"     # one sentence, then exit
    python main.py --model llama-base               # which of the four models
    python main.py --stages "Budak2 tu gilaaa!!"    # show the cleaning steps
    python main.py --raw "Budak2 tu gilaaa!!"       # skip cleaning entirely
    python main.py --plain "budak2 tu gila"         # lemma sentence only
    python main.py --lang en "budak2 tu gila"       # English rule glosses

THE ARCHITECTURE THIS FILE IS
-----------------------------
    user input
       |
       v  preprocess.py     normalise -> eliminate -> tokenise
       v  skill.py          the system prompt, byte-identical to training
       v  the selected LLM  one of the four in MODELS below
       v  skill.py          parse the answer, render the derivation
       v
    final output

Each stage lives in its own module and is runnable on its own. This file is
only the wiring and the REPL -- if something here looks like linguistics, it
is in the wrong file.

WHAT YOU GET BACK
-----------------
The lemma sentence, then one line per token: the token, its lemma, and the
rule the model says produced it. `python skill.py --rules` prints the nine
rules; `python skill.py --card` prints the instructions the model is given,
including the JANGAN list of things it must not do.

The rule tags are the MODEL'S OWN CLAIM about what it did, not a re-derivation
made here. That is the point of the reasoning format: when a lemma is wrong,
the tag tells you which rule misfired.

PREPROCESSING IS ON BY DEFAULT, AND THAT IS A CHOICE
----------------------------------------------------
Input is cleaned before it reaches the model. Whether that helps depends on
whether the training targets were cleaned the same way -- if the model was
fine-tuned on raw corpus text carrying capitals, commas and digits, then
cleaning here shifts the input away from what it saw in training. Use --raw to
send the sentence through untouched, and compare, before trusting either.
"""

import argparse
import sys
from pathlib import Path

import lexicon
import skill
from preprocess import NUMBER_POLICIES, preprocess, trace

# ---------------------------------------------------------------------------
# THE FOUR MODELS
# ---------------------------------------------------------------------------
#
# Two base models and their two fine-tunes. The bases are matched on purpose:
# both ~3B, both instruction-tuned, both in the same 4-bit Unsloth packaging,
# so the only thing that differs between the families is the pretraining.
#
# Every one of them is served the SAME system prompt. A base model that cannot
# hold the output format under the instructions the fine-tune was trained on
# is a result worth reporting, not a handicap to correct for.

ROOT = Path(__file__).resolve().parent

# The two base models are Hugging Face ids. The two fine-tunes are LOCAL
# folders, and are resolved to absolute paths rather than left as
# "unsloth/lemmatizer_lora_llama" -- which reads as a Hugging Face repo under
# the `unsloth` org, because that org really exists. A relative path here
# would also break the moment main.py is run from anywhere but the project
# root. Both failures look like a download error, not a path error.
BASE_MODELS = {
    "llama-base": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    "qwen-base":  "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
}
ADAPTERS = {
    "llama-ft": ROOT / "unsloth" / "lemmatizer_lora_llama",
    "qwen-ft":  ROOT / "unsloth" / "lemmatizer_lora_qwen",
}
MODELS = dict(BASE_MODELS, **ADAPTERS)
DEFAULT_MODEL = "llama-ft"

# Unsloth's chat-template name per family. Getting this wrong does not raise:
# it silently wraps the prompt in the wrong special tokens, and the model
# answers fluent nonsense -- the same class of failure as a stale system
# prompt, and just as hard to spot.
CHAT_TEMPLATES = (("qwen", "qwen-2.5"), ("llama", "llama-3.2"))
DEFAULT_TEMPLATE = "llama-3.2"

# The reasoning format is a long answer: ~15 lines of derivation plus the
# sentence, against the ~30 tokens a bare lemma sentence needs. Sized so a
# 30-token input cannot truncate its own derivation mid-table, which reads as
# a model that cannot finish a thought.
MAX_SEQ_LEN = 1024
MAX_NEW_TOKENS = 512


def resolve_model(name):
    """A --model value -> (reference to load, display name).

    Accepts a key from MODELS, a Hugging Face id, or a path to an adapter
    folder, so a newly trained adapter can be tried without editing this file.
    """
    if name in MODELS:
        return MODELS[name], name
    return name, name


def template_for(ref):
    """The chat template this model family needs."""
    low = str(ref).lower()
    for needle, template in CHAT_TEMPLATES:
        if needle in low:
            return template
    print("[main] cannot tell which family %r belongs to - falling back to %s. "
          "If the answers look like fluent nonsense, this is why."
          % (str(ref), DEFAULT_TEMPLATE))
    return DEFAULT_TEMPLATE


# ---------------------------------------------------------------------------
# THE MODEL
# ---------------------------------------------------------------------------

def load_model(ref):
    """Load a model, its tokenizer, and the system prompt it expects.

    unsloth and torch are imported HERE rather than at module scope so that
    --help, --stages and every preprocessing path stay runnable on a machine
    with no GPU. Importing unsloth costs several seconds and a CUDA probe;
    nothing above this line needs either.
    """
    try:
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
    except ImportError:
        sys.exit("unsloth is not installed, so no model can be loaded.\n"
                 "The preprocessing stages still work without it:\n"
                 "    python preprocess.py --demo\n"
                 "    python main.py --no-model \"Budak2 tu gilaaa!!\"")

    # A base model was never fine-tuned, so it carries no skill stamp to read.
    # Everything else is asked, never assumed -- a model served a prompt it was
    # not trained on still answers fluently, just in the wrong format, and the
    # score drops for a reason that has nothing to do with the weights.
    if str(ref) in BASE_MODELS.values():
        system_prompt = skill.SYSTEM_PROMPT
        version = skill.SKILL_VERSION + " (base, untrained)"
    else:
        system_prompt, version = skill.prompt_for(str(ref))

    print("Loading %s (skill %s)..." % (ref, version))
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(ref),
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template=template_for(ref))
    FastLanguageModel.for_inference(model)

    # These models ship generation_config.max_length = 131072 and we pass
    # max_new_tokens instead. transformers warns about the pair on every
    # generate() call, which puts a paragraph of noise above each answer.
    # max_new_tokens already wins; clearing the unused one only quiets it.
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.max_length = None

    print("Ready.\n")
    return model, tokenizer, system_prompt


def lemmatise(model, tokenizer, text, system_prompt):
    """One sentence -> the model's raw answer.

    Greedy decoding, not sampling. Lemmatisation has one right answer per
    token, so there is nothing for temperature to buy -- and a reported score
    that changes between runs of the same model on the same data cannot be
    checked by anyone.
    """
    msgs = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": text}]

    inputs = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0][inputs.shape[1]:],
                            skip_special_tokens=True)


# ---------------------------------------------------------------------------
# INPUT AND OUTPUT
# ---------------------------------------------------------------------------

def prepare(text, clean=True, numbers="strip", stages=False):
    """Run the preprocessing, optionally showing its working."""
    if not clean:
        return text
    if stages:
        width = max(len(n) for n, _ in trace(text, numbers))
        for name, value in trace(text, numbers):
            if isinstance(value, list):
                value = "[%s]" % ", ".join(repr(t) for t in value)
            print("  %-*s  %s" % (width, name, value))
        print()
    return preprocess(text, numbers)


def render(sent, response, lang="ms", plain=False, use_lexicon=True):
    """Print the answer.

    An answer that will not parse is printed RAW rather than swallowed. An
    unreadable answer is a result too: it is how you find out that a base
    model cannot hold the format, which is one of the things the four-way
    comparison exists to measure.
    """
    parsed = skill.parse_response(response)

    # Applied AFTER parsing, so the model still sees and answers the raw rojak
    # it was trained on; only its answer is corrected. Every correction is
    # attributed below rather than folded into the model's own output -- the
    # derivation format exists so a reader can tell which component produced
    # each lemma, and a silent override would destroy that.
    n_fixed = lexicon.apply(parsed) if use_lexicon else 0

    if plain:
        print(parsed["sentence"] or "(no answer)")
        return parsed

    if not parsed["steps"]:
        print("  the model did not produce a parseable derivation.")
        print("  raw answer:\n")
        for line in response.strip().splitlines():
            print("    " + line)
        print()
        return parsed

    print(skill.format_result(parsed, lang))
    if n_fixed:
        print()
        print("  %d token%s corrected from the lexicon, not by the model:"
              % (n_fixed, "" if n_fixed == 1 else "s"))
        for s in parsed["steps"]:
            if s.get("from_lexicon"):
                print("    %-14s model said %-14s lexicon says %s"
                      % (s["token"], s["model_lemma"], s["lemma"]))
    print()
    return parsed


# ---------------------------------------------------------------------------
# THE REPL
# ---------------------------------------------------------------------------

BANNER = """\
Malay rojak lemmatiser. Type a sentence, or:
  /raw      toggle preprocessing on/off      /stages   toggle the cleaning trace
  /plain    toggle lemma-sentence-only       /lang     toggle ms/en glosses
  /help     show this again                  quit      exit (or exit, q, Ctrl-C)
"""

# Accepted bare, without a leading slash, because that is what anyone who has
# used a REPL types first. The cost is that the English words "exit" and "quit"
# can no longer be lemmatised on their own -- type them inside a longer
# sentence, or as "/exit", if you ever need to.
QUIT_WORDS = {"quit", "exit", "q", "/quit", "/exit", "/q"}


def repl(model, tokenizer, system_prompt, clean, numbers, stages, plain,
         lang, use_lexicon=True):
    print(BANNER)
    while True:
        try:
            text = input("rojak> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return

        if not text:
            continue
        if text.lower() in QUIT_WORDS:
            print("Goodbye!")
            return
        if text == "/help":
            print(BANNER)
            continue
        if text == "/raw":
            clean = not clean
            print("preprocessing %s\n" % ("ON" if clean else "OFF"))
            continue
        if text == "/stages":
            stages = not stages
            print("cleaning trace %s\n" % ("ON" if stages else "OFF"))
            continue
        if text == "/plain":
            plain = not plain
            print("plain output %s\n" % ("ON" if plain else "OFF"))
            continue
        if text == "/lang":
            lang = "en" if lang == "ms" else "ms"
            print("glosses: %s\n" % lang)
            continue
        if text.startswith("/"):
            print("unknown command %r - /help for the list\n" % text)
            continue

        # Per turn, not per session. A CUDA OOM on one long sentence, or a
        # decode error on one odd input, should cost you that answer and not
        # the loaded model -- reloading 3B weights to recover from a typo is a
        # bad trade, and the session state (toggles, history) goes with it.
        try:
            sent = prepare(text, clean, numbers, stages)
            if not sent:
                print("  nothing left after cleaning - try /raw\n")
                continue
            render(sent, lemmatise(model, tokenizer, sent, system_prompt),
                   lang, plain, use_lexicon)
        except KeyboardInterrupt:
            # Interrupting a slow generation returns you to the prompt; it does
            # not exit, because Ctrl-C at the prompt itself already does that.
            print("\n  (interrupted)\n")
        except Exception as exc:                                # noqa: BLE001
            print("  error: %s: %s\n" % (type(exc).__name__, exc))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    # Rojak input carries emoji, and the default Windows console codepage
    # raises on them mid-print rather than at the boundary.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="models: %s (or any HF id / adapter path)"
               % ", ".join(MODELS))
    p.add_argument("sentence", nargs="*", help="lemmatise this, then exit")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="which model (default %s)" % DEFAULT_MODEL)
    p.add_argument("--raw", action="store_true",
                   help="send the sentence to the model uncleaned")
    p.add_argument("--stages", action="store_true",
                   help="show each preprocessing stage")
    p.add_argument("--no-lexicon", action="store_true",
                   help="do not correct the model's answer from the lexicon; "
                        "show what the model alone produces")
    p.add_argument("--no-model", action="store_true",
                   help="preprocess only and stop - no GPU, no model load")
    p.add_argument("--plain", action="store_true",
                   help="print the lemma sentence only, no derivation")
    p.add_argument("--lang", default="ms", choices=("ms", "en"),
                   help="language of the rule glosses (default ms)")
    p.add_argument("--numbers", default="strip", choices=NUMBER_POLICIES,
                   help="digit handling in preprocessing (default strip)")
    a = p.parse_args()
    clean = not a.raw

    # Inspecting the preprocessing is a legitimate thing to do on a machine
    # with no GPU, so it must not pay for a model load. Returns before unsloth
    # is ever imported.
    if a.no_model:
        if not a.sentence:
            sys.exit("--no-model needs a sentence to preprocess.")
        text = " ".join(a.sentence)
        out = prepare(text, clean, a.numbers, stages=True)
        print("  -> to the model: %r" % out)
        return

    ref, _name = resolve_model(a.model)
    model, tokenizer, system_prompt = load_model(ref)

    if a.sentence:
        text = " ".join(a.sentence)
        sent = prepare(text, clean, a.numbers, a.stages)
        if not sent:
            sys.exit("nothing left after cleaning - try --raw")
        render(sent, lemmatise(model, tokenizer, sent, system_prompt),
               a.lang, a.plain, not a.no_lexicon)
    else:
        repl(model, tokenizer, system_prompt, clean, a.numbers,
             a.stages, a.plain, a.lang, not a.no_lexicon)


if __name__ == "__main__":
    main()
