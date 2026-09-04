import argparse
import sys
from pathlib import Path

import lexicon
import skill
from preprocess import NUMBER_POLICIES, preprocess, trace


ROOT = Path(__file__).resolve().parent

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


CHAT_TEMPLATES = (("qwen", "qwen-2.5"), ("llama", "llama-3.2"))
DEFAULT_TEMPLATE = "llama-3.2"

MAX_SEQ_LEN = 1024
MAX_NEW_TOKENS = 512


def resolve_model(name):
    if name in MODELS:
        return MODELS[name], name
    return name, name


def template_for(ref):
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
    try:
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
    except ImportError:
        sys.exit("unsloth is not installed, so no model can be loaded.\n"
                 "The preprocessing stages still work without it:\n"
                 "    python preprocess.py --demo\n"
                 "    python main.py --no-model \"Budak2 tu gilaaa!!\"")
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

    if getattr(model, "generation_config", None) is not None:
        model.generation_config.max_length = None

    print("Ready.\n")
    return model, tokenizer, system_prompt


def lemmatise(model, tokenizer, text, system_prompt):
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
    parsed = skill.parse_response(response)
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


BANNER = """\
Malay rojak lemmatiser. Type a sentence, or:
  /raw      toggle preprocessing on/off      /stages   toggle the cleaning trace
  /plain    toggle lemma-sentence-only       /lang     toggle ms/en glosses
  /help     show this again                  quit      exit (or exit, q, Ctrl-C)
"""

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



def main():
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
