"""Lemmatise text with the rules, the fine-tuned model, or both.

Each sentence goes to the model in the same form the training data used, and
with follow_rules=True every token the rules are sure about keeps the rule
lemma - the exact label it would have had in the training data. The model
decides the rest (ambiguous words, affixes, unknown spellings).
"""

# pyarrow must be loaded before torch on Windows, or importing unsloth
# crashes the process with no error message
import pyarrow  # noqa: F401

import difflib
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import rules                                                      # noqa: E402
import skill                                                      # noqa: E402
from preprocess import prepare                                    # noqa: E402

MODELS_DIR = ROOT / "models"

BASE_MODELS = {
    "llama-base": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    "qwen-base": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
}
ADAPTERS = {
    "llama-ft": MODELS_DIR / "lemmatizer_lora_llama",
    "qwen-ft": MODELS_DIR / "lemmatizer_lora_qwen",
}
CHAT_TEMPLATES = (("qwen", "qwen-2.5"), ("llama", "llama-3.2"))

MAX_SEQ_LEN = 1024
MAX_NEW_TOKENS = 512


def gpu_packages():
    spec = importlib.util.find_spec("unsloth")
    return spec is not None and spec.origin is not None


def adapters_on_disk():
    return [name for name, path in ADAPTERS.items()
            if (path / "adapter_model.safetensors").exists()]


def resolve(name):
    if name in ADAPTERS:
        return ADAPTERS[name]
    if name in BASE_MODELS:
        return BASE_MODELS[name]
    path = Path(name)
    return path if path.exists() else name


def template_for(ref):
    low = str(ref).lower()
    for needle, template in CHAT_TEMPLATES:
        if needle in low:
            return template
    meta = Path(ref) / "adapter_config.json"
    if meta.exists() and "qwen" in meta.read_text(encoding="utf-8").lower():
        return "qwen-2.5"
    return "llama-3.2"


class Model:
    def __init__(self, name):
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template

        ref = resolve(name)
        if str(ref) in BASE_MODELS.values():
            self.prompt, self.version = skill.SYSTEM_PROMPT, skill.SKILL_VERSION
        else:
            self.prompt, self.version = skill.prompt_for(ref)

        model, tok = FastLanguageModel.from_pretrained(
            model_name=str(ref), max_seq_length=MAX_SEQ_LEN,
            dtype=None, load_in_4bit=True)
        tok = get_chat_template(tok, chat_template=template_for(ref))
        FastLanguageModel.for_inference(model)
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.max_length = None

        self.name, self.model, self.tok = name, model, tok

    def generate(self, sentence):
        msgs = [{"role": "system", "content": self.prompt},
                {"role": "user", "content": sentence}]
        import torch
        ids = self.tok.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True,
            return_tensors="pt").to(self.model.device)
        out = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                  max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                  pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

    def unload(self):
        import gc
        self.model = self.tok = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def rule_steps(sentence):
    steps = []
    for e in rules.label(sentence):
        tag, unsure = skill.rule_for(e["src"])
        steps.append({"token": e["t"], "lemma": e["l"], "rule": tag,
                      "uncertain": unsure, "src": e["src"], "by": "rules"})
    return steps


def _same(a, b, src):
    return a == b if src == "name" else a.lower() == b.lower()


def merge(steps, parsed, follow_rules=True):
    """Line the model's derivation up with the rule tokens and combine them."""
    ours = [s["token"].lower() for s in steps]
    theirs = [s["token"].lower() for s in parsed["steps"]]
    matched = {}
    sm = difflib.SequenceMatcher(None, ours, theirs, autojunk=False)
    for block in sm.get_matching_blocks():
        for k in range(block.size):
            matched[block.a + k] = parsed["steps"][block.b + k]

    out = []
    for i, r in enumerate(steps):
        m = matched.get(i)
        if m is None:
            out.append(dict(r, note="model skipped this token"))
        elif follow_rules and r["src"] in rules.CERTAIN \
                and not _same(m["lemma"], r["lemma"], r["src"]):
            out.append(dict(r, model_lemma=m["lemma"]))
        else:
            out.append(dict(r, lemma=m["lemma"], rule=m["rule"],
                            uncertain=m["uncertain"], by="model"))
    return out


def lemmatise(text, model=None, follow_rules=True):
    """One result per sentence: the lemma sentence plus a step per token."""
    results = []
    for sent in prepare(text):
        steps = rule_steps(sent)
        raw, ok = None, None
        if model is not None:
            raw = model.generate(sent)
            parsed = skill.parse_response(raw)
            ok = parsed["ok"]
            steps = merge(steps, parsed, follow_rules)
        results.append({
            "input": sent,
            "steps": steps,
            "sentence": " ".join(s["lemma"] for s in steps),
            "raw": raw,
            "ok": ok,
        })
    return results
