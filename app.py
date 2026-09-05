#!/usr/bin/env python
"""
Malay Rojak Lemmatiser - interactive demo.

    streamlit run app.py

Same pipeline as main.py, with the stages made visible: preprocess -> model
-> lexicon corrections -> lemma sentence plus the derivation behind it.

Serves one system, the fine-tuned Llama selected on the validation set; run
another by hand with main.py --model qwen-ft (or llama-base, qwen-base).

Not named streamlit.py on purpose: a module of that name in the project root
shadows the streamlit package itself.
"""

# Must stay FIRST, above anything that can pull in torch: Arrow loaded after
# torch corrupts the process heap on Windows and kills the server (0xC0000374,
# no traceback) on the first st.dataframe render. Do not move or regroup.
import pyarrow  # noqa: F401  (imported for its side effect: load order)

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import lexicon                                                  # noqa: E402
import main as cli                                              # noqa: E402
import skill                                                    # noqa: E402
from preprocess import NUMBER_POLICIES, trace                   # noqa: E402

SELECTED = "llama-ft"
BLURB = "Fine-tuned Llama 3.2 3B + LoRA - selected on the validation set"


@st.cache_resource(show_spinner=False)
def gpu_pool():
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="lemmatiser-gpu")


def on_gpu(fn, *args, **kwargs):
    return gpu_pool().submit(fn, *args, **kwargs).result()


def _free_gb():
    import torch
    if not torch.cuda.is_available():
        return None
    free, _total = torch.cuda.mem_get_info()
    return free / 1e9


def gpu_free_gb():
    try:
        return on_gpu(_free_gb)
    except Exception:                                           # noqa: BLE001
        return None


def model_error(exc, name):
    msg = str(exc)
    if ("dispatched on the CPU" in msg or "out of memory" in msg.lower()
            or "OutOfMemory" in type(exc).__name__):
        st.error("Not enough GPU memory to load **%s**." % name)
        st.caption("A 3B model in 4-bit needs about 2.2 GB and this card "
                   "has 4 GB, so only one fits at a time. Press **Unload "
                   "model** in the sidebar, then try again. If nothing is "
                   "loaded, close any other process holding the GPU "
                   "(a training run, another Streamlit tab, or main.py).")
        unload()
    else:
        st.error("Could not run the model: %s: %s" % (type(exc).__name__, msg))


def show_status(slot):
    held = st.session_state.get("model_choice")
    free = gpu_free_gb()
    with slot.container():
        st.caption("Loaded: **%s**" % held if held else "Not loaded")
        if free is not None:
            st.caption("GPU free: %.1f GB" % free)


def _to_cpu(bundle):
    try:
        bundle[0].cpu()
    except Exception:                                           # noqa: BLE001
        # a quantised model that refuses to move is still worth collecting
        pass


def _sweep():
    import gc

    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def unload():
    bundle = st.session_state.pop("model_bundle", None)
    st.session_state.pop("model_choice", None)
    if bundle is None:
        return

    try:
        on_gpu(_to_cpu, bundle)
    except Exception:                                           # noqa: BLE001
        pass
    del bundle          # this frame's reference, so the sweep can free the blocks

    try:
        on_gpu(_sweep)
    except Exception:                                           # noqa: BLE001
        pass


def load(choice):
    if st.session_state.get("model_choice") == choice:
        return st.session_state["model_bundle"]

    if st.session_state.get("model_choice"):
        with st.spinner("Unloading %s..." % st.session_state["model_choice"]):
            unload()

    ref, _ = cli.resolve_model(choice)
    with st.spinner("Loading %s (~40s)..." % choice):
        model, tok, prompt = on_gpu(cli.load_model, ref)

    bundle = (model, tok, prompt)
    st.session_state["model_bundle"] = bundle
    st.session_state["model_choice"] = choice
    return bundle


st.set_page_config(page_title="Malay Rojak Lemmatiser", layout="wide")
st.title("Malay Rojak Lemmatiser")
st.caption("Type informal Malaysian Malay. The model returns the lemma form "
           "and shows which rule produced each one.")

with st.sidebar:
    st.header("Model")
    st.caption(BLURB)

    on_disk = (SELECTED in cli.BASE_MODELS
               or (cli.MODELS[SELECTED] / "adapter_model.safetensors").exists())
    if not on_disk:
        st.error("%s is not on disk. Train it with `python train.py --model "
                 "llama`, or copy the adapter folder into unsloth/."
                 % SELECTED)

    status = st.empty()

    if st.session_state.get("model_choice") \
            and st.button("Unload model", use_container_width=True):
        unload()
        st.session_state["hold_off"] = SELECTED
        st.rerun()

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

ready = True
held_off = st.session_state.get("hold_off") == SELECTED
if not held_off:
    try:
        load(SELECTED)
    except Exception as exc:                                    # noqa: BLE001
        model_error(exc, SELECTED)
        ready = False

show_status(status)
if held_off:
    st.info("**%s** is unloaded and the GPU is free. Press **Lemmatise** to "
            "load it again." % SELECTED)

with st.form("lemmatise"):
    text = st.text_area("Sentence", height=90,
                        placeholder="aku dh xleh handle keje tu la weh")
    go = st.form_submit_button("Lemmatise", type="primary", disabled=not ready)

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
        st.session_state.pop("hold_off", None)
        model, tok, prompt = load(SELECTED)
        with st.spinner("Lemmatising..."):
            raw = on_gpu(cli.lemmatise, model, tok, sent, prompt)
    except Exception as exc:                                    # noqa: BLE001
        model_error(exc, SELECTED)
        if stages:
            with st.expander("Preprocessing still ran", expanded=True):
                for name, value in stages:
                    st.text("%-16s %s" % (name, value))
        st.stop()

    parsed = skill.parse_response(raw)
    n_fixed = lexicon.apply(parsed) if use_lex else 0

    if not parsed["steps"]:
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
