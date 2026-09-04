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

The page serves ONE system: the fine-tuned Llama 3.2 3B selected on the
validation set. There is no model picker, because a demonstration should show
what the project recommends rather than offer a forty-second model swap in
front of whoever is watching. The four-way comparison belongs to the
evaluation, which already reports it; to run another model by hand, use
main.py --model qwen-ft (or llama-base, qwen-base).

NOT named streamlit.py on purpose: a module of that name in the project root
shadows the streamlit package itself, and `import streamlit` then finds the
project file instead of the library.
"""

# pyarrow is imported FIRST, before anything can pull in torch, and the order
# is the whole point of the line -- st.dataframe serialises through Arrow, and
# Arrow loaded after torch corrupts the process heap on Windows. The symptom is
# brutal to read: the lemma renders correctly, then the server dies with
# exception 0xC0000374 and no Python traceback, so the browser shows only
# "Connection error. Is Streamlit still running?" and the crash looks like it
# came from the model rather than from drawing the results table.
#
#     import order  torch -> pyarrow   first table render kills the server
#     import order  pyarrow -> torch   survives repeated runs
#
# Do not move this below the other imports, and do not "tidy" it into the
# third-party group.
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

# The one system this page serves. `llama-ft` is the model the selection
# protocol chose on the validation set: statistically tied with the fine-tuned
# Qwen, and taken on the pre-declared tie-break of format-failure rate then
# character error rate.
#
# Changing this string is all it takes to demonstrate a different system --
# any key of main.MODELS works -- but the page shows one at a time, because a
# 3B model in 4-bit needs about 2.2 GB and the card has 4 GB.
SELECTED = "llama-ft"
BLURB = "Fine-tuned Llama 3.2 3B + LoRA - selected on the validation set"


# ---------------------------------------------------------------------------
# ONE THREAD OWNS THE GPU
# ---------------------------------------------------------------------------
#
# Streamlit runs every rerun of this script on a NEW thread and lets that
# thread die when the run ends. Touching CUDA from a thread that then exits
# corrupts the process heap: the rerun immediately after a generation dies with
# Windows exception 0xC0000374 (STATUS_HEAP_CORRUPTION) and no Python
# traceback. The browser reports that as a connection error, so the app looks
# like it disconnected when really the server process is gone -- which is why
# the answer appears and the page drops a moment later.
#
# Measured on this machine with no Streamlit involved at all, so this is a
# CUDA/thread-lifetime problem that Streamlit's execution model merely trips:
#
#     generate on a thread, join it, start any further thread   -> 0xC0000374
#     generate on ONE worker thread that never exits            -> survives
#
# So every CUDA call -- loading, generating, unloading, even reading free VRAM
# for the sidebar -- is submitted to a single worker thread that lives as long
# as the process. Script threads come and go without ever touching the driver.
# The pool is deliberately never shut down.


@st.cache_resource(show_spinner=False)
def gpu_pool():
    """The one thread allowed to talk to CUDA. Cached, so created once."""
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="lemmatiser-gpu")


def on_gpu(fn, *args, **kwargs):
    """Run fn on the GPU thread and wait for the answer.

    Call this from the script thread, and never call st.* inside fn: widgets
    need the script thread's context, which the worker does not have. Wrap the
    call in st.spinner instead -- the spinner is entered here, the work happens
    over there.
    """
    return gpu_pool().submit(fn, *args, **kwargs).result()


def _free_gb():
    import torch
    if not torch.cuda.is_available():
        return None
    free, _total = torch.cuda.mem_get_info()
    return free / 1e9


def gpu_free_gb():
    """Free VRAM in GB, or None if there is no GPU to ask.

    Reading it is a CUDA call, so it goes through the worker like everything
    else. The figure is what the DRIVER has left: PyTorch's caching allocator
    keeps whatever it has taken, so after a generation this reads near zero
    even though the model itself needs only about 2.2 GB.
    """
    try:
        return on_gpu(_free_gb)
    except Exception:                                           # noqa: BLE001
        return None


def model_error(exc, name):
    """Explain a load or generate failure in terms of this card.

    Shared by the start-up load and the generate call so that the same fault
    is not described two different ways depending on when it happened.
    """
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
    """Fill the sidebar's reserved slot with what is actually loaded now.

    Called after the load, not before it, because loading is what changes the
    answer. Writing into an st.empty() replaces what was there rather than
    appending, so the slot never shows two contradicting captions.
    """
    held = st.session_state.get("model_choice")
    free = gpu_free_gb()
    with slot.container():
        st.caption("Loaded: **%s**" % held if held else "Not loaded")
        if free is not None:
            st.caption("GPU free: %.1f GB" % free)


def _to_cpu(bundle):
    """Move the weights off the card. Runs on the GPU thread."""
    try:
        bundle[0].cpu()
    except Exception:                                           # noqa: BLE001
        # A quantised model that refuses to move is still worth collecting;
        # the caller will find out from the free figure.
        pass


def _sweep():
    """Collect, then hand the freed blocks back to the driver."""
    import gc

    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def unload():
    """Drop the loaded model and give its VRAM back.

    WHY THIS EXISTS AT ALL
    ----------------------
    A 3B model in 4-bit occupies roughly 2.2 GB, so a 4 GB card holds exactly
    one, and the page holds it for as long as it is open. Unload is how the
    card is handed back to something else -- a training run, or main.py -- and
    also the recovery path when a second process has already taken the memory:
    bitsandbytes reports that as "Some modules are dispatched on the CPU or the
    disk", which reads like a configuration problem rather than the memory
    exhaustion it is.

    THE ORDER MATTERS, AND .cpu() IS NOT OPTIONAL
    ---------------------------------------------
    Dropping the reference and calling empty_cache() is NOT enough: unsloth
    patches modules globally and keeps its own references to the loaded model,
    so the weights stay resident and the next load fails. Measured on this
    card, 4 GB total:

        loaded                    1.03 GB free
        del + gc + empty_cache    1.03 GB free   <- nothing released
        .cpu() first, then those  3.39 GB free   <- released

    Moving the weights to host memory releases the CUDA blocks whether or not
    anything else still holds a Python reference, and empty_cache() then
    returns them to the driver.
    """
    bundle = st.session_state.pop("model_bundle", None)
    st.session_state.pop("model_choice", None)
    if bundle is None:
        return

    try:
        on_gpu(_to_cpu, bundle)
    except Exception:                                           # noqa: BLE001
        pass
    # Dropped here as well: the worker released its own reference, but this
    # frame still held one, and the sweep below is what returns the blocks.
    del bundle
    try:
        on_gpu(_sweep)
    except Exception:                                           # noqa: BLE001
        pass


def load(choice):
    """The requested model, loading it only if it is not already the one held.

    Kept in session_state rather than @st.cache_resource so that it can be
    EVICTED: Streamlit's cache has no eviction hook, so a cached model stays
    resident on the GPU for the life of the process, and the Unload button
    could not give the card back. The parameter stays even though the page
    serves one model, so that pointing SELECTED at another key is the only
    edit needed.

    The loading itself is main.load_model, not a copy of it, so the GUI and the
    CLI get the same weights, the same per-family chat template and the same
    system prompt. A model served the wrong template does not raise -- it
    answers fluently in the wrong format, which reads as a weak model rather
    than a misconfigured one, so this is worth having in exactly one place.

    It runs on the GPU thread for the reason given above the pool: a model
    loaded from a thread that later exits takes the process down with it.

    What is deliberately NOT done anywhere here is sys.modules.pop("unsloth").
    Forcing a re-import re-runs unsloth's global patching, and the second model
    load in a process then dies without a traceback -- the interpreter simply
    stops. Importing once and reusing the cached module loads model after
    model in one process without incident.
    """
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
        st.error("%s is not on disk. Run import_adapters.py to add it."
                 % SELECTED)

    # Reserved now, written once the run knows what is actually loaded. The
    # sidebar is drawn before the model is, so filling this in here would
    # report the state the page started in rather than the state it ends in.
    status = st.empty()

    if st.session_state.get("model_choice") \
            and st.button("Unload model", use_container_width=True):
        # The page loads the model on every run, so unloading alone would be
        # undone by the very next run and the only visible effect would be a
        # wasted forty seconds. Recording the dismissal suppresses that reload
        # until the model is actually asked for again by pressing Lemmatise.
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

# The model is loaded when the page opens, not on the first submit. The wait is
# the same either way, but where it falls changes how it reads: forty seconds
# after pressing Lemmatise looks like the app has hung, while forty seconds
# during start-up looks like start-up. By the time the box can be typed into,
# the model behind it is resident.
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

# A form, not a bare text_area plus button. Streamlit commits a text_area only
# when it loses focus, so typing and clicking in one motion sends the click
# before the text and the first press appears to do nothing. Inside a form both
# arrive together, and Ctrl+Enter submits.
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
        # Normally already resident, loaded above before the form was drawn.
        # The exception is a model dismissed with Unload: asking for it again
        # is what brings it back, so the hold is lifted here.
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
