#!/usr/bin/env python
"""
Malay Rojak Lemmatiser - side-by-side demo.

    streamlit run app.py

Shows the same sentence through three systems at once, because the headline
result of this project is a comparison, not a single number:

    MALINDO      dictionary lookup alone. What main.py does today, and what
                 a Malay lemmatiser gives you off the shelf. Scores 0.8% on
                 the tokens that actually need changing.
    RULES        MALINDO root column + slang lexicon + orthographic repair.
                 The strongest baseline buildable without machine learning.
    FINE-TUNE    the QLoRA adapter. Optional - the page works without a GPU,
                 and only loads the model when you ask it to.

NOT named streamlit.py on purpose: a module of that name in the project root
shadows the streamlit package itself, and `import streamlit` then finds the
project file instead of the library.
"""

import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "crawl"))
sys.path.insert(0, str(ROOT / "unsloth"))

from make_goldsheet import (WORD_RE, malindo_lemmas,        # noqa: E402
                            malindo_roots, pick, propose)

ADAPTER_ROOT = ROOT / "unsloth"
SYSTEM_PROMPT = ("Anda adalah pelemma bahasa Melayu. Tukarkan ayat rojak "
                 "kepada ayat berbentuk lema. Kembalikan ayat sahaja.")

# Adapters deliberately not offered in the picker. yt8k trains on the same
# corpus with a different label convention and scores lower on the measures
# that matter (79.8% vs 86.2% on tokens needing change); it stays on disk for
# the report comparison but is not something to demo.
HIDDEN_ADAPTERS = {"lemmatizer_lora_yt8k"}


# ---------------------------------------------------------------------------
# BACKENDS
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading MALINDO Morph...")
def load_lexicons():
    return malindo_lemmas(), malindo_roots()


def run_malindo(tokens, lex, roots):
    """Dictionary lookup only - no slang lexicon, no orthographic repair."""
    out = []
    for t in tokens:
        lemma, _ = pick(lex.get(t, []), t)
        out.append(lemma or t)
    return out


def run_rules(tokens, lex, roots):
    return [propose(t, lex, roots)[0] for t in tokens]


def rule_tags(tokens, lex, roots):
    return [propose(t, lex, roots)[1] for t in tokens]


def _ensure_cc():
    """Point triton at its bundled TinyCC.

    Windows Store Python reports a site-packages path that packages are not
    actually installed into, so triton cannot find the compiler it ships with
    and every kernel build fails. Setting CC here means the page works without
    the user having to remember an environment variable.
    """
    if os.environ.get("CC"):
        return
    try:
        import triton
        tcc = Path(triton.__file__).parent / "runtime" / "tcc" / "tcc.exe"
        if tcc.exists():
            os.environ["CC"] = str(tcc)
    except Exception:
        pass


@st.cache_resource(show_spinner="Loading the fine-tuned model (~30s)...")
def load_model(adapter_name):
    """Import unsloth without the project's own `unsloth/` folder shadowing it."""
    _ensure_cc()
    shadow = {"", ".", str(ROOT)}
    saved = list(sys.path)
    sys.path[:] = [p for p in sys.path if p not in shadow]
    sys.modules.pop("unsloth", None)
    try:
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
        model, tok = FastLanguageModel.from_pretrained(
            model_name=str(ADAPTER_ROOT / adapter_name),
            max_seq_length=512, dtype=None, load_in_4bit=True)
        tok = get_chat_template(tok, chat_template="llama-3.2")
        FastLanguageModel.for_inference(model)
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.max_length = None
        return model, tok
    finally:
        sys.path[:] = saved


def run_model(model, tok, text):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}]
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                  return_tensors="pt").to("cuda")
    out = model.generate(input_ids=ids, max_new_tokens=200, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# PAGE
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Malay Rojak Lemmatiser", layout="wide")
st.title("Malay Rojak Lemmatiser")
st.caption("Dictionary vs rules vs fine-tuned model, on the same sentence.")

adapters = sorted(p.name for p in ADAPTER_ROOT.glob("lemmatizer_lora*")
                  if (p / "adapter_model.safetensors").exists()
                  and p.name not in HIDDEN_ADAPTERS)

with st.sidebar:
    st.header("Fine-tuned model")
    if adapters:
        adapter = st.selectbox("Adapter", adapters,
                               index=adapters.index("lemmatizer_lora_yt")
                               if "lemmatizer_lora_yt" in adapters else 0)
        use_model = st.checkbox("Enable (needs GPU)", value=False)
    else:
        adapter, use_model = None, False
        st.info("No adapter found under unsloth/.")

lex, roots = load_lexicons()

text = st.text_area(
    "Sentence",
    height=80,
    placeholder="aku dh xleh handle keje tu la weh",
)

if st.button("Lemmatise", type="primary") and text.strip():
    tokens = WORD_RE.findall(text.lower())
    if not tokens:
        st.warning("Nothing to lemmatise.")
        st.stop()

    md = run_malindo(tokens, lex, roots)
    rl = run_rules(tokens, lex, roots)
    tags = rule_tags(tokens, lex, roots)

    rows = []
    for i, t in enumerate(tokens):
        rows.append({"token": t, "MALINDO": md[i], "rules": rl[i],
                     "rule applied": tags[i],
                     "dictionary missed it": "yes" if md[i] != rl[i] else ""})
    st.subheader("Token by token")
    st.dataframe(rows, use_container_width=True, hide_index=True)

    missed = sum(1 for i in range(len(tokens)) if md[i] != rl[i])
    st.caption("The dictionary left %d of %d tokens unchanged that the rule "
               "pipeline normalised." % (missed, len(tokens)))

    a, b = st.columns(2)
    a.markdown("**MALINDO only**")
    a.info(" ".join(md))
    b.markdown("**Rules**")
    b.success(" ".join(rl))

    if use_model and adapter:
        try:
            model, tok = load_model(adapter)
            st.markdown("**Fine-tuned model** (`%s`)" % adapter)
            st.success(run_model(model, tok, text))
        except Exception as e:
            st.error("Could not run the model: %s" % e)
            st.caption("The dictionary and rule columns above are unaffected.")
