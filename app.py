#!/usr/bin/env python
"""
Malay Rojak Lemmatiser - interactive demo.

    streamlit run app.py

The same pipeline main.py runs, with the intermediate stages made visible:

    your sentence
       |
       v  preprocess.py   normalise -> eliminate -> tokenise -> expand slang
       v  the model       one derivation line per token
       v  lexicon.py      dictionary corrections, labelled as such
       v
    lemma sentence + the derivation behind it

Defaults to the selected model, the fine-tuned Llama 3.2 3B, but any of the
four evaluated systems can be picked from the sidebar. Loading a base model
alongside the fine-tune is the quickest way to see what fine-tuning bought.

NOT named streamlit.py on purpose: a module of that name in the project root
shadows the streamlit package itself, and `import streamlit` then finds the
project file instead of the library.
"""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import lexicon                                                  # noqa: E402
import main as cli                                              # noqa: E402
import skill                                                    # noqa: E402
from preprocess import NUMBER_POLICIES, trace                   # noqa: E402

# Sidebar order: the selected model first, then its sibling, then the two
# untrained baselines. A demo should open on the system the project actually
# recommends, not on whatever sorts first alphabetically.
CHOICES = ["llama-ft", "qwen-ft", "llama-base", "qwen-base"]
BLURB = {
    "llama-ft":   "Fine-tuned Llama 3.2 3B - the selected model",
    "qwen-ft":    "Fine-tuned Qwen 2.5 3B - statistically tied with Llama",
    "llama-base": "Llama 3.2 3B, no fine-tuning",
    "qwen-base":  "Qwen 2.5 3B, no fine-tuning",
}


@st.cache_resource(show_spinner="Loading the model (~40s the first time)...")
def load(choice):
    """Load a model once per session and keep it.

    unsloth is imported HERE, not at module scope, so the page renders and the
    preprocessing panel works on a machine with no GPU. The sys.path dance is
    because this project has its own `unsloth/` directory: with the project
    root on the path, `import unsloth` can resolve to that folder as an empty
    namespace package, and every attribute lookup then fails with an error
    that says nothing about the real cause.
    """
    ref, _ = cli.resolve_model(choice)

    shadow = {"", ".", str(ROOT)}
    saved = list(sys.path)
    sys.path[:] = [p for p in sys.path if p not in shadow]
    sys.modules.pop("unsloth", None)
    try:
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template

        model, tok = FastLanguageModel.from_pretrained(
            model_name=str(ref), max_seq_length=cli.MAX_SEQ_LEN,
            dtype=None, load_in_4bit=True)
        # Per family. A base model has no skill stamp to read, and serving any
        # model the wrong template does not raise -- it answers fluently in the
        # wrong format, which looks like a weak model rather than a
        # misconfigured one.
        tok = get_chat_template(tok, chat_template=cli.template_for(ref))
        FastLanguageModel.for_inference(model)
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.max_length = None

        if str(ref) in cli.BASE_MODELS.values():
            prompt = skill.SYSTEM_PROMPT
        else:
            prompt, _v = skill.prompt_for(str(ref))
        return model, tok, prompt
    finally:
        sys.path[:] = saved


st.set_page_config(page_title="Malay Rojak Lemmatiser", layout="wide")
st.title("Malay Rojak Lemmatiser")
st.caption("Type informal Malaysian Malay. The model returns the lemma form "
           "and shows which rule produced each one.")

with st.sidebar:
    st.header("Model")
    available = [c for c in CHOICES
                 if c in cli.BASE_MODELS
                 or (cli.MODELS[c] / "adapter_model.safetensors").exists()]
    missing = [c for c in CHOICES if c not in available]

    choice = st.selectbox("System", available,
                          format_func=lambda c: "%s  -  %s" % (c, BLURB[c]))
    if missing:
        st.caption("Not on disk: %s. Run import_adapters.py to add them."
                   % ", ".join(missing))

    st.header("Pipeline")
    clean = st.checkbox("Preprocess the input", value=True,
                        help="Off sends the sentence exactly as typed, which "
                             "is the form the model was fine-tuned on.")
    numbers = st.selectbox("Digits", NUMBER_POLICIES, index=0,
                           help="strip keeps the letters around a digit "
                                "(8pagi -> pagi); drop removes the whole token.")
    use_lex = st.checkbox("Apply the lexicon", value=True,
                          help="Corrects the model's answer from "
                               "data/user_lexicon.csv. Corrections are shown "
                               "separately, never folded into the model's own "
                               "output.")
    lang = st.radio("Rule glosses", ("ms", "en"), horizontal=True)

# A form, not a bare text_area plus button. Streamlit commits a text_area only
# when it loses focus, so typing and clicking in one motion sends the click
# before the text and the first press appears to do nothing. Inside a form both
# arrive together, and Ctrl+Enter submits.
with st.form("lemmatise"):
    text = st.text_area("Sentence", height=90,
                        placeholder="aku dh xleh handle keje tu la weh")
    go = st.form_submit_button("Lemmatise", type="primary")

if go:
    if not text.strip():
        st.warning("Type a sentence first.")
        st.stop()

    stages = trace(text, numbers) if clean else None
    sent = stages[-1][1] if stages else None
    sent = " ".join(sent) if isinstance(sent, list) else (sent or text)
    if not sent.strip():
        st.warning("Nothing survived preprocessing - try letters rather than "
                   "only digits, emoji or punctuation.")
        st.stop()

    try:
        model, tok, prompt = load(choice)
        raw = cli.lemmatise(model, tok, sent, prompt)
    except Exception as exc:                                    # noqa: BLE001
        st.error("Could not run the model: %s: %s" % (type(exc).__name__, exc))
        if stages:
            with st.expander("Preprocessing still ran", expanded=True):
                for name, value in stages:
                    st.text("%-16s %s" % (name, value))
        st.stop()

    parsed = skill.parse_response(raw)
    n_fixed = lexicon.apply(parsed) if use_lex else 0

    if not parsed["steps"]:
        # An unreadable answer is a result, not an error to swallow: it is how
        # a base model's inability to hold the format shows itself.
        st.error("The model did not produce a parseable derivation.")
        st.caption("Expected for an un-fine-tuned base model: on the gold test "
                   "set base Llama failed this on 39.3% of inputs.")
        st.code(raw or "(empty)")
        st.stop()

    st.subheader("Lemma sentence")
    st.success(parsed["sentence"])

    st.subheader("How each token was resolved")
    gloss = skill.RULES_EN if lang == "en" else skill.RULES
    rows = []
    for s in parsed["steps"]:
        rows.append({
            "token": s["token"],
            "lemma": s["lemma"],
            "rule": s["rule"] + ("?" if s.get("uncertain") else ""),
            "what that means": gloss.get(s["rule"], ""),
            "source": "lexicon" if s.get("from_lexicon") else "model",
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

    changed = sum(1 for s in parsed["steps"]
                  if s["lemma"].lower() != s["token"].lower())
    st.caption("%d of %d tokens were changed; the rest were already in lemma "
               "form." % (changed, len(parsed["steps"])))

    if n_fixed:
        with st.expander("%d token%s corrected from the lexicon, not by the "
                         "model" % (n_fixed, "" if n_fixed == 1 else "s"),
                         expanded=True):
            st.caption("Shown separately so the derivation stays an account "
                       "of what the model did.")
            st.dataframe(
                [{"token": s["token"], "model said": s["model_lemma"],
                  "lexicon says": s["lemma"]}
                 for s in parsed["steps"] if s.get("from_lexicon")],
                use_container_width=True, hide_index=True)

    if stages:
        with st.expander("Preprocessing stages"):
            for name, value in stages:
                st.text("%-16s %s" % (name, value))
    else:
        st.caption("Preprocessing off - the sentence went to the model as "
                   "typed.")

    with st.expander("What the model actually returned"):
        st.code(raw)
