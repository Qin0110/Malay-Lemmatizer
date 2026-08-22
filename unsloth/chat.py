#!/usr/bin/env python
"""
Try the Malay rojak lemmatiser on your own sentences.

    python chat.py                                   # interactive
    python chat.py "aku dh xleh handle keje tu"      # one sentence, then exit
    python chat.py --adapter lemmatizer_lora_yt      # pick which model
    python chat.py --stages "Budak2 tu gilaaa!!"     # show the cleaning steps
    python chat.py --raw "Budak2 tu gilaaa!!"        # skip cleaning entirely

There is more than one adapter in this repo and they are NOT
interchangeable - each was trained with its own system prompt and target
format. Defaults to lemmatizer_lora; pass --adapter for the others.

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

# Must match training config in llm.py
ADAPTER_DIR = "lemmatizer_lora"
MAX_SEQ_LEN = 512
CHAT_TEMPLATE = "llama-3.2"

# CRITICAL: must be BYTE-IDENTICAL to the prompt used in training.
# Imported rather than retyped - this file previously carried a stale copy
# asking for "token -> lema" line format, which the model was never trained
# on and which made every answer nonsense.
from llm import SYSTEM_PROMPT      # noqa: E402

# Stdlib-only and GPU-free, so it stays testable on its own:
#     python preprocess.py --demo
from preprocess import preprocess, trace, NUMBER_POLICIES   # noqa: E402


def load_model(adapter_dir=ADAPTER_DIR):
    """Load the fine-tuned model and tokenizer."""
    print("Loading model from", adapter_dir)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_dir,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template=CHAT_TEMPLATE)
    FastLanguageModel.for_inference(model)
    print("Model loaded successfully!\n")
    return model, tokenizer


def lemmatize(model, tokenizer, text):
    """Generate lemmas for the input text."""
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
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
        max_new_tokens=256,
        do_sample=False,  # greedy decoding for deterministic results
        pad_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(
        outputs[0][inputs.shape[1]:],
        skip_special_tokens=True
    )

    return response


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


def interactive_mode(adapter_dir=ADAPTER_DIR, clean=True,
                     numbers="strip", stages=False):
    """Run interactive chat loop."""
    model, tokenizer = load_model(adapter_dir)

    print("=" * 70)
    print("Malay Rojak Lemmatizer - Interactive Mode")
    print("=" * 70)
    print("\nEnter Malay text to lemmatize (or 'quit' to exit)")
    print("Preprocessing:", "off (raw input)" if not clean
          else "on, numbers=%s" % numbers)
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
            result = lemmatize(model, tokenizer, sent)
            print(result)
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
    a = p.parse_args()

    if a.sentence:
        text = " ".join(a.sentence)
        model, tokenizer = load_model(a.adapter)
        print()
        print("in  :", text)
        sent = prepare_input(text, not a.raw, a.numbers, a.stages)
        if not sent:
            raise SystemExit("nothing left after preprocessing")
        print("out :", lemmatize(model, tokenizer, sent).strip())
    else:
        interactive_mode(a.adapter, not a.raw, a.numbers, a.stages)
