#!/usr/bin/env python
"""
Try the Malay rojak lemmatiser on your own sentences.

    python chat.py                                   # interactive
    python chat.py "aku dh xleh handle keje tu"      # one sentence, then exit
    python chat.py --adapter lemmatizer_lora_yt      # pick which model

There is more than one adapter in this repo and they are NOT
interchangeable - each was trained with its own system prompt and target
format. Defaults to lemmatizer_lora; pass --adapter for the others.
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


def interactive_mode(adapter_dir=ADAPTER_DIR):
    """Run interactive chat loop."""
    model, tokenizer = load_model(adapter_dir)

    print("=" * 70)
    print("Malay Rojak Lemmatizer - Interactive Mode")
    print("=" * 70)
    print("\nEnter Malay text to lemmatize (or 'quit' to exit)")
    print("\nExample: aku dh xleh handle kerja tu la weh\n")

    while True:
        try:
            text = input("\n> ").strip()

            if not text:
                continue

            if text.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            print("\nLemmatizing...\n")
            result = lemmatize(model, tokenizer, text)
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
    a = p.parse_args()

    if a.sentence:
        text = " ".join(a.sentence)
        model, tokenizer = load_model(a.adapter)
        print()
        print("in  :", text)
        print("out :", lemmatize(model, tokenizer, text).strip())
    else:
        interactive_mode(a.adapter)
