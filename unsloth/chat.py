#!/usr/bin/env python
"""
Try the Malay rojak lemmatiser on your own sentences.

    python chat.py                                   # interactive
    python chat.py "aku dh xleh handle keje tu"      # one sentence, then exit
    python chat.py --adapter lemmatizer_lora_yt      # pick which model
    python chat.py --stages "Budak2 tu gilaaa!!"     # show the cleaning steps
    python chat.py --raw "Budak2 tu gilaaa!!"        # skip cleaning entirely
    python chat.py --plain "budak2 tu gila"          # lemma sentence only
    python chat.py --lang en "budak2 tu gila"        # English rule glosses
    python chat.py --skill v2 "budak2 tu gila"       # force the other prompt

There is more than one adapter in this repo and they are NOT interchangeable -
each was trained with its own system prompt and target format. Which prompt an
adapter needs is recorded in skill.ADAPTERS and asked for by name, so you no
longer have to remember. Defaults to lemmatizer_lora_yt.

WHAT YOU GET BACK
-----------------
One line per token: the token, its lemma, the rule that produced it, and what
that rule means. The nine rules are skill.RULES; `python skill.py --rules`
prints them, `python skill.py --card` prints the instructions the model is
given, including the JANGAN list of things it must not do.

Where those rule tags come from depends on the adapter, and the difference
matters:

  v2 adapters state the rule themselves - it is part of what they were trained
    to output, and the tag is the model's own claim about what it did.
  v1 adapters (lemmatizer_lora, lemmatizer_lora_yt) answer with a bare lemma
    sentence. The tags are re-derived here from the token/lemma pairs by
    skill.derive_steps(), and marked "[nota]" so nobody mistakes them for the
    model's reasoning. Use --skill v2 on a v1 adapter to see why prompting
    alone does not fix this.

Input is cleaned by preprocess.py before it reaches the model: lowercase and
elongation, then reduplication, then emoji/symbol/digit removal, then
tokenisation. Use --raw to send the sentence through untouched, which is the
only way to compare the two fairly.

NOTE: the adapters in this repo were fine-tuned on UNCLEANED text -- the user
turns in data/training.jsonl still carry capitals, commas and digits. So
cleaning the input here shifts it away from what the model saw in training.
It is still usually the right call for messy input, but the honest fix is to
run the same preprocessing over the training targets and retrain, so the two
sides match. Compare with --raw before trusting either.
"""

import argparse

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

# Must match training config in llm.py. lemmatizer_lora and
# lemmatizer_lora_yt8k were removed from this repo, so the only adapter left
# is the one this pipeline is about.
ADAPTER_DIR = "lemmatizer_lora_yt"
CHAT_TEMPLATE = "llama-3.2"

# Per skill version, because v2 is a bigger job in both directions: its targets
# run 150-300 tokens against v1's ~30, and its system prompt is far longer.
# Loading a v2 adapter at v1's 512 would truncate the derivation mid-answer,
# which looks like a model that cannot finish a thought.
MAX_SEQ_LEN = {"v1": 512, "v2": 1024}
MAX_NEW_TOKENS = {"v1": 256, "v2": 512}

# CRITICAL: the system prompt must be BYTE-IDENTICAL to the one used in
# training. Asked for by adapter, never retyped: this file once carried a stale
# copy asking for a "token -> lema" line format the model had never been
# trained on, and every answer came out nonsense.
#
# Importing it from llm.py fixed the typo but not the bug. llm.py picks v1 or
# v2 off the USE_REASONING environment variable, which describes a TRAINING
# run, not the adapter being loaded - so with that variable unset (default
# "1"), `--adapter lemmatizer_lora_yt` was handed the v2 reasoning prompt for a
# v1 model. skill.prompt_for() asks the adapter instead, which is the only
# thing that actually knows.
import skill                       # noqa: E402

# Stdlib-only and GPU-free, so it stays testable on its own:
#     python preprocess.py --demo
from preprocess import preprocess, trace, NUMBER_POLICIES   # noqa: E402


def resolve_skill(adapter_dir, override="auto"):
    """(system prompt, version) for this run.

    `override` exists for one experiment worth being able to run: point the v2
    prompt at a v1 adapter and watch the output format collapse. That is the
    evidence that the reasoning format has to be trained in and cannot be
    prompted in, so it is worth making easy rather than hiding.
    """
    prompt, version = skill.prompt_for(adapter_dir)
    if override != "auto" and override != version:
        print("[skill] adapter %s expects %s; forcing %s. Expect the output "
              "format to break - that is the point of this flag."
              % (adapter_dir, version, override))
        return skill.PROMPTS[override], override
    return prompt, version


def load_model(adapter_dir=ADAPTER_DIR, override="auto"):
    """Load the fine-tuned model, its tokenizer, and the skill it speaks."""
    system_prompt, version = resolve_skill(adapter_dir, override)
    print("Loading model from %s (skill %s)" % (adapter_dir, version))
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_dir,
        max_seq_length=MAX_SEQ_LEN[version],
        dtype=None,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template=CHAT_TEMPLATE)
    FastLanguageModel.for_inference(model)

    # Llama 3.2 ships generation_config.max_length = 131072 and we pass
    # max_new_tokens instead. transformers 5.x warns about the pair on every
    # generate() call, which puts a paragraph of noise above each answer.
    # max_new_tokens already wins; clearing the unused one only quiets it.
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.max_length = None

    print("Model loaded successfully!\n")
    return model, tokenizer, system_prompt, version


def lemmatize(model, tokenizer, text, system_prompt, version="v1"):
    """Generate lemmas for the input text."""
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]

    inputs = tokenizer.apply_chat_template(
        msgs,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=MAX_NEW_TOKENS[version],
        do_sample=False,  # greedy decoding for deterministic results
        pad_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(
        outputs[0][inputs.shape[1]:],
        skip_special_tokens=True
    )

    return response


def render(sent, response, version, lang="ms", plain=False):
    """Print the answer: the derivation table, then the lemma sentence.

    A v2 answer that will not parse is printed raw rather than swallowed. An
    unreadable answer is a result too - it is how you find out the adapter and
    the prompt disagree - and hiding it behind "(tiada analisis)" would cost
    you the one clue on screen.
    """
    response = response.strip()
    if plain:
        print(response)
        return

    if version == "v2":
        parsed = skill.parse_response(response)
        if not parsed["steps"]:
            print(response)
            return
    else:
        parsed = skill.derive_steps(sent.split(), response)

    print(skill.format_result(parsed, lang))


def prepare_input(text, clean=True, numbers="strip", stages=False):
    """Run the preprocessing pipeline and show what it did.

    Returns the string to send to the model, or "" if nothing survived. An
    input of pure emoji, digits or punctuation cleans down to nothing, and an
    empty user turn does not produce an empty answer - it makes the model
    invent a sentence to lemmatise. Better to say so and ask again.
    """
    if not clean:
        return text

    if stages:
        for name, value in trace(text, numbers):
            print("  %-16s %s" % (name, value))

    sent = preprocess(text, numbers)
    if not stages and sent != text:
        print("clean:", sent)
    return sent


def interactive_mode(adapter_dir=ADAPTER_DIR, clean=True, numbers="strip",
                     stages=False, lang="ms", plain=False, override="auto"):
    """Run interactive chat loop."""
    model, tokenizer, system_prompt, version = load_model(adapter_dir, override)

    print("=" * 70)
    print("Malay Rojak Lemmatizer - Interactive Mode")
    print("=" * 70)
    print("\nEnter Malay text to lemmatize (or 'quit' to exit)")
    print("Preprocessing:", "off (raw input)" if not clean
          else "on, numbers=%s" % numbers)
    print("Skill:", version, "- run `python skill.py --card` to see the rules")
    print("\nExample: Budak2 tu gilaaa!! 3 org je\n")

    while True:
        try:
            text = input("\n> ").strip()

            if not text:
                continue

            if text.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            sent = prepare_input(text, clean, numbers, stages)
            if not sent:
                print("\nNothing left after preprocessing - try again.")
                continue

            print("\nLemmatizing...\n")
            result = lemmatize(model, tokenizer, sent, system_prompt, version)
            render(sent, result, version, lang, plain)
            print()

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("sentence", nargs="*",
                   help="lemmatise this and exit; omit for interactive mode")
    p.add_argument("--adapter", default=ADAPTER_DIR,
                   help="adapter folder to load (default %s)" % ADAPTER_DIR)
    p.add_argument("--raw", action="store_true",
                   help="skip preprocessing, send the sentence as typed")
    p.add_argument("--stages", action="store_true",
                   help="print every preprocessing stage")
    p.add_argument("--numbers", default="strip", choices=NUMBER_POLICIES,
                   help="what preprocessing does with digits (default: strip)")
    p.add_argument("--lang", default="both", choices=("ms", "en", "both"),
                   help="rule-gloss language. Default 'both': tags as "
                        "RINGKAS/SHORT with English descriptions. Use 'ms' for "
                        "Malay-only output.")
    p.add_argument("--plain", action="store_true",
                   help="print the model's answer only, no derivation table")
    p.add_argument("--skill", default="auto", choices=("auto", "v1", "v2"),
                   help="override the prompt version (default: ask the adapter)")
    a = p.parse_args()

    if a.sentence:
        text = " ".join(a.sentence)
        model, tokenizer, system_prompt, version = load_model(a.adapter, a.skill)
        print()
        print("in  :", text)
        sent = prepare_input(text, not a.raw, a.numbers, a.stages)
        if not sent:
            raise SystemExit("nothing left after preprocessing")
        print()
        render(sent, lemmatize(model, tokenizer, sent, system_prompt, version),
               version, a.lang, a.plain)
    else:
        interactive_mode(a.adapter, not a.raw, a.numbers, a.stages,
                         a.lang, a.plain, a.skill)
