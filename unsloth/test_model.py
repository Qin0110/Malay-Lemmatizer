"""
Test / chat with the fine-tuned Malay rojak lemmatizer.
 
    python test_model.py --chat        # interactive loop
    python test_model.py --probe       # generalisation diagnostics
    python test_model.py --compare     # base model vs fine-tune, side by side
 
Output is a LEMMA SENTENCE, matching build_500.py / train_lemmatizer.py.
SYSTEM_PROMPT must stay byte-identical across all three files.
"""
 
import argparse
 
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import TextStreamer
 
ADAPTER_DIR = "lemmatizer_lora"
BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
MAX_SEQ_LEN = 512
 
# MUST be byte-identical to the string used during training.
SYSTEM_PROMPT = (
    "Anda adalah pelemma bahasa Melayu. Tukarkan ayat rojak kepada ayat "
    "berbentuk lema. Kembalikan ayat sahaja."
)
 
_model = _tokenizer = None
 
 
def load(path=ADAPTER_DIR):
    global _model, _tokenizer
    _model, _tokenizer = FastLanguageModel.from_pretrained(
        model_name=path,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
    )
    _tokenizer = get_chat_template(_tokenizer, chat_template="llama-3.2")
    FastLanguageModel.for_inference(_model)      # ~2x faster generation
    return _model, _tokenizer
 
 
def lemmatize(text, stream=False, max_new_tokens=160):
    if _model is None:
        load()
 
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    inputs = _tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")
 
    streamer = TextStreamer(_tokenizer, skip_prompt=True) if stream else None
    out = _model.generate(
        input_ids=inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,                 # deterministic task: never sample
        streamer=streamer,
        pad_token_id=_tokenizer.eos_token_id,
    )
    return _tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()
 
 
def chat():
    """Interactive loop. Type 'quit' to exit."""
    if _model is None:
        load()
    print("Rojak lemmatizer ready. Type a sentence, or 'quit'.\n")
    while True:
        try:
            text = input("rojak> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"quit", "exit", "q"}:
            break
        if not text:
            continue
        print()
        lemmatize(text, stream=True)
        print()
 
 
# ---------------------------------------------------------------------------
# Diagnostics: does it generalise, or did it memorise?
# ---------------------------------------------------------------------------
 
PROBES = [
    # Seen in training - should be near-perfect. If these fail, training failed.
    ("SEEN", "aku dh xleh handle kerja tu la weh"),
 
    # Unseen slang, same pattern as trained forms
    ("UNSEEN SLANG", "die tgk movie tu smlm"),
    ("UNSEEN SLANG", "npe ko tak dtg semalam"),
 
    # Elongation with a length never seen in training
    ("ELONGATION", "wahhh sedapppp giler makanan ni"),
 
    # Reduplication on unseen stems
    ("REDUPLICATION", "kereta2 tu semua dah rosak"),
 
    # English words absent from the training set
    ("CODESWITCH", "aku kena debug code ni sebelum deploy"),
 
    # Derived Malay forms not in the lexicon
    ("MORPHOLOGY", "penyelidikan itu memerlukan pembiayaan tambahan"),
 
    # Time/money/date formats in unseen shapes
    ("NUMERIC", "harga naik RM2300 sejak 15/8/2026 pukul 7am"),
 
    # Longer than any training sentence
    ("LENGTH", "semalam aku pergi kedai dengan kawan aku sebab dia nak "
               "beli barang untuk majlis hari jadi adik dia minggu depan"),
 
    # Off-task: a narrow fine-tune should still try to lemmatise this
    ("OFF-TASK", "What is the capital of Malaysia?"),
]
 
 
def probe():
    if _model is None:
        load()
    for tag, text in PROBES:
        print("=" * 68)
        print(f"[{tag}]  {text}\n")
        print(lemmatize(text))
        print()
 
 
def compare():
    """Base model vs fine-tune on the same prompts. This is your before/after."""
    print("### BASE MODEL (no fine-tune) ###\n")
    load(BASE_MODEL)
    for tag, text in PROBES[:5]:
        print(f"[{tag}] {text}\n{lemmatize(text)}\n{'-' * 60}")
 
    print("\n### FINE-TUNED ###\n")
    load(ADAPTER_DIR)
    for tag, text in PROBES[:5]:
        print(f"[{tag}] {text}\n{lemmatize(text)}\n{'-' * 60}")
 
 
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--chat", action="store_true")
    p.add_argument("--probe", action="store_true")
    p.add_argument("--compare", action="store_true")
    a, _ = p.parse_known_args()
 
    if a.chat:
        chat()
    elif a.probe:
        probe()
    elif a.compare:
        compare()
    else:
        probe()