#!/usr/bin/env python
# pyarrow has to be imported before torch on Windows, or the first table
# render can crash the server
import pyarrow  # noqa: F401

import csv
import html
import io
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pipeline                                                 # noqa: E402
import skill                                                    # noqa: E402
from preprocess import trace                                    # noqa: E402

RULES_ONLY = "Rules only (no GPU)"
REPO = "https://github.com/lgw123456789/Malay-Rojak-Lemmatiser"
BATCH_LIMIT = {"rules": 2000, "model": 200}

EXAMPLES = [
    "Aku dh xleh handle keje tu, Kak Uji pun tau",
    "Safwan best gilaaa!! nnti aku share link kat korang",
    "Budak2 tu mmg x reti duduk diam la weh",
    "Sy nk tny sikit, kedai tu bukak x arini?",
]

COLOURS = {
    "RINGKAS":  "245, 158, 11",
    "IMBUHAN":  "139, 92, 246",
    "GANDA":    "236, 72, 153",
    "PANJANG":  "6, 182, 212",
    "PARTIKEL": "148, 163, 184",
    "NAMA":     "34, 197, 94",
    "INGGERIS": "59, 130, 246",
    "NOMBOR":   "100, 116, 139",
    "KAMUS":    "120, 120, 120",
    "KEKAL":    "239, 68, 68",
}

CHIP_CSS = """
<style>
.chips {display: flex; flex-wrap: wrap; gap: 6px; margin: 4px 0 12px;}
.chip {display: inline-flex; flex-direction: column; align-items: center;
       padding: 4px 10px; border-radius: 8px; border: 1px solid;
       line-height: 1.3; min-width: 2.5em;}
.chip .t {font-size: 0.75em; opacity: 0.75;}
.chip .l {font-weight: 600;}
.chip .r {font-size: 0.6em; letter-spacing: 0.05em; opacity: 0.8;}
.chip.unsure {border-style: dashed;}
.chip.fixed {box-shadow: 0 0 0 2px rgba(239, 68, 68, 0.6);}
.legend {display: flex; flex-wrap: wrap; gap: 6px; font-size: 0.8em;}
.legend span {padding: 1px 8px; border-radius: 6px; border: 1px solid;}
</style>
"""


@st.cache_resource(show_spinner=False)
def gpu_worker():
    # one thread owns the model, so CUDA always sees the same thread
    return ThreadPoolExecutor(max_workers=1)


def on_gpu(fn, *args):
    return gpu_worker().submit(fn, *args).result()


def get_model(name):
    current = st.session_state.get("model")
    if current is not None and current.name == name:
        return current
    if current is not None:
        on_gpu(current.unload)
        st.session_state.pop("model", None)
    with st.spinner("Loading %s (about 30-60 s the first time)..." % name):
        model = on_gpu(pipeline.Model, name)
    st.session_state["model"] = model
    return model


def model_error(exc):
    msg = str(exc)
    if isinstance(exc, ImportError):
        st.error("The model packages are not installed. Run `SETUP.bat gpu`, "
                 "or use **%s**." % RULES_ONLY)
    elif "out of memory" in msg.lower() or "dispatched on the CPU" in msg:
        st.error("Not enough GPU memory for the model. Close other programs "
                 "using the GPU and try again.")
    else:
        st.error("Could not run the model: %s: %s" % (type(exc).__name__, msg))


def run(text, engine, follow):
    """Lemmatise with the chosen engine, loading the model if needed."""
    if engine == RULES_ONLY:
        return pipeline.lemmatise(text, None, follow), None
    model = get_model(engine)
    return on_gpu(pipeline.lemmatise, text, model, follow), model


def chip(step):
    rgb = COLOURS.get(step["rule"], COLOURS["KAMUS"])
    classes = ["chip"]
    if step["uncertain"]:
        classes.append("unsure")
    if step.get("model_lemma"):
        classes.append("fixed")
    tip = "%s: %s" % (step["rule"], skill.RULES_EN.get(step["rule"], ""))
    if step.get("model_lemma"):
        tip += " | model said %s, training rules kept %s" % (step["model_lemma"], step["lemma"])
    return ('<span class="%s" style="background: rgba(%s, 0.15); border-color: rgba(%s, 0.7)" '
            'title="%s"><span class="t">%s</span><span class="l">%s</span>'
            '<span class="r">%s</span></span>'
            % (" ".join(classes), rgb, rgb, html.escape(tip, quote=True),
               html.escape(step["token"]), html.escape(step["lemma"]), step["rule"]))


def legend():
    return '<div class="legend">%s</div>' % "".join(
        '<span style="background: rgba(%s, 0.15); border-color: rgba(%s, 0.7)" title="%s">%s</span>'
        % (rgb, rgb, html.escape(skill.RULES_EN[tag]), tag)
        for tag, rgb in COLOURS.items())


def table_rows(results, gloss):
    rows = []
    for n, res in enumerate(results, 1):
        for s in res["steps"]:
            rows.append({
                "sentence": n,
                "token": s["token"],
                "lemma": s["lemma"],
                "rule": s["rule"] + ("?" if s["uncertain"] else ""),
                "meaning": gloss.get(s["rule"], ""),
                "decided by": s["by"],
                "model said": s.get("model_lemma", ""),
            })
    return rows


def to_csv(rows):
    buf = io.StringIO()
    if rows:
        w = csv.DictWriter(buf, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return buf.getvalue().encode("utf-8-sig")


def use_example(text):
    st.session_state["text"] = text


def read_batch(upload, pasted):
    lines = []
    if upload is not None:
        raw = upload.getvalue().decode("utf-8-sig", errors="replace")
        if upload.name.lower().endswith(".csv"):
            reader = csv.reader(io.StringIO(raw))
            rows = list(reader)
            if rows and rows[0] and rows[0][0].strip().lower() in ("text", "sentence", "ayat"):
                rows = rows[1:]
            lines = [r[0] for r in rows if r]
        else:
            lines = raw.splitlines()
    lines += pasted.splitlines()
    return [l.strip() for l in lines if l.strip()]


def show_single(text, engine, follow, gloss):
    if not text.strip():
        st.warning("Type a sentence first.")
        return
    try:
        with st.spinner("Lemmatising..."):
            results, model = run(text, engine, follow)
    except Exception as exc:                                    # noqa: BLE001
        model_error(exc)
        return
    if not results:
        st.warning("Nothing left after cleaning - try words rather than "
                   "only emoji, links or symbols.")
        return

    steps = [s for res in results for s in res["steps"]]
    changed = sum(1 for s in steps if s["lemma"].lower() != s["token"].lower())
    names = sum(1 for s in steps if s["rule"] == "NAMA")
    fixed = sum(1 for s in steps if s.get("model_lemma"))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Words", len(steps))
    m2.metric("Changed", changed)
    m3.metric("Proper nouns", names)
    m4.metric("Model answers corrected", fixed if model is not None else "-",
              help="Words where the model disagreed with the training-data "
                   "label and the label was kept")

    st.subheader("Lemma sentence")
    st.code(" ".join(r["sentence"] for r in results), language=None, wrap_lines=True)

    st.subheader("Word by word")
    st.markdown('<div class="chips">%s</div>' % "".join(chip(s) for s in steps),
                unsafe_allow_html=True)
    st.caption("Hover over a word for details. Dashed border: the rules were "
               "not sure. Red outline: the model's answer was replaced by "
               "the training-data label.")
    with st.expander("Colour key"):
        st.markdown(legend(), unsafe_allow_html=True)

    rows = table_rows(results, gloss)
    st.dataframe(rows, width="stretch", hide_index=True)
    st.download_button("Download as CSV", to_csv(rows),
                       file_name="lemmas.csv", mime="text/csv")

    if any(r["ok"] is False for r in results):
        st.warning("The model's answer was not fully in the expected "
                   "format; the rule labels filled the gaps.")

    with st.expander("How the input was cleaned"):
        for name, value in trace(text):
            st.text("%-10s %s" % (name, value))
    if model is not None:
        with st.expander("Raw model output"):
            for res in results:
                st.code(res["raw"] or "(empty)", language=None)


def show_batch(lines, engine, follow, gloss):
    limit = BATCH_LIMIT["rules" if engine == RULES_ONLY else "model"]
    if not lines:
        st.warning("No sentences found.")
        return
    if len(lines) > limit:
        st.warning("Only the first %d lines are processed with this engine." % limit)
        lines = lines[:limit]

    out, detail = [], []
    bar = st.progress(0.0, text="0 / %d" % len(lines))
    try:
        for i, line in enumerate(lines, 1):
            results, _ = run(line, engine, follow)
            out.append({"input": line,
                        "lemmas": " ".join(r["sentence"] for r in results)})
            for row in table_rows(results, gloss):
                row["sentence"] = i
                detail.append(row)
            bar.progress(i / len(lines), text="%d / %d" % (i, len(lines)))
    except Exception as exc:                                    # noqa: BLE001
        model_error(exc)
        return
    bar.empty()

    st.dataframe(out, width="stretch", hide_index=True)
    c1, c2 = st.columns(2)
    c1.download_button("Download sentences (CSV)", to_csv(out),
                       file_name="lemmas_sentences.csv", mime="text/csv")
    c2.download_button("Download word by word (CSV)", to_csv(detail),
                       file_name="lemmas_words.csv", mime="text/csv")



st.set_page_config(page_title="Malay Rojak Lemmatiser", page_icon="📝", layout="wide")
st.markdown(CHIP_CSS, unsafe_allow_html=True)
st.title("Malay Rojak Lemmatiser")
st.caption("Type informal Malaysian Malay and get the lemma of every word, "
           "with the rule behind it.")

with st.sidebar:
    st.header("Engine")
    choices = [RULES_ONLY]
    if pipeline.gpu_packages():
        choices += pipeline.adapters_on_disk() + list(pipeline.BASE_MODELS)
    engine = st.selectbox("Lemmatiser", choices,
                          index=1 if len(choices) > 1 and choices[1].endswith("-ft") else 0)
    if not pipeline.gpu_packages():
        st.caption("Install the GPU packages (`SETUP.bat gpu`) to use the "
                   "fine-tuned model.")
    elif not pipeline.adapters_on_disk():
        st.caption("No fine-tuned adapter in `models/` yet - run "
                   "`python download_model.py`.")

    follow = st.checkbox(
        "Follow the training rules exactly", value=True,
        help="Words the rules are sure about (slang, particles, names, "
             "numbers...) keep the label they had in the training data. "
             "The model decides the rest.")
    lang = st.radio("Rule glosses", ("ms", "en"), horizontal=True)

    status = st.empty()

    st.divider()
    st.caption("[Source code and documentation](%s)" % REPO)

gloss = skill.RULES_EN if lang == "en" else skill.RULES
single, batch = st.tabs(["One sentence", "Many sentences"])

with single:
    st.caption("Try an example:")
    cols = st.columns(len(EXAMPLES))
    for col, example in zip(cols, EXAMPLES):
        col.button(example, on_click=use_example, args=(example,), width="stretch")

    with st.form("lemmatise"):
        text = st.text_area("Sentence", key="text", height=90,
                            placeholder="Aku dh xleh handle keje tu, Kak Uji pun tau")
        go = st.form_submit_button("Lemmatise", type="primary")
    if go:
        show_single(text, engine, follow, gloss)

with batch:
    st.caption("One sentence per line. Upload a .txt file (or a .csv whose "
               "first column is the text), or paste the lines below.")
    upload = st.file_uploader("File", type=["txt", "csv"])
    pasted = st.text_area("Or paste sentences", height=150, key="batch_text")
    if st.button("Lemmatise all", type="primary"):
        show_batch(read_batch(upload, pasted), engine, follow, gloss)

# filled in last, so it already shows a model that was loaded during this run
with status.container():
    loaded = st.session_state.get("model")
    if loaded is not None:
        st.success("Loaded: **%s**" % loaded.name)
        if st.button("Unload model", width="stretch"):
            on_gpu(loaded.unload)
            st.session_state.pop("model", None)
            st.rerun()
    elif engine != RULES_ONLY:
        st.info("The model loads when you press Lemmatise.")
