# Command reference

Every command in the pipeline, in the order it runs. Anything marked **GPU**
loads a 3B model; everything else runs on CPU.

On Windows use the venv interpreter explicitly, because the system Python has
CPU-only torch and fails with a confusing error:

```
E:/Malay_Lemmatizer/venv/Scripts/python.exe
```

The commands below say `python` for brevity. Substitute that path.

---

## 0. Setup — a teammate starting from a clone

The repo does not contain the model weights: a LoRA adapter is 110–126 MB and
GitHub rejects any file over 100 MB. Get `adapters.zip` separately, extract it
anywhere, then:

```bash
python import_adapters.py --from "C:/Users/you/Downloads/adapters" --dry-run
python import_adapters.py --from "C:/Users/you/Downloads/adapters"
```

`--from` points at the extracted **folder**, not the `.zip`. The dry run
verifies both adapters carry the same step count before anything is copied — a
mismatch would mean the two models saw different amounts of data, and the
comparison between them would measure schedule rather than pretraining.

Without the weights everything except generation still works: preprocessing,
the rule baselines, split verification and all the data scripts.

---

## 1. Data collection — already done, do not re-run

These built the corpus. Re-running them would produce different sentence ids
and orphan the 300 hand-labelled sentences, which cost hours to produce and
cannot be recovered. They are documented for the report, not for use.

```bash
python crawl/youtube_crawl.py --list          # show configured channels
python crawl/youtube_crawl.py --probe         # sample channels, save nothing
python crawl/youtube_crawl.py --videos 25     # the real crawl (needs an API key in .env)

python crawl/build_corpus.py                  # raw comments -> data/corpus.jsonl
python crawl/build_corpus.py --validate       # also write a hand-check sample
python crawl/score_filter.py                  # precision/recall of the rojak filter

python crawl/build_silver.py --n 8000         # corpus -> silver + the 90/10 split
python crawl/build_silver.py --resplit        # redo the split without relabelling
```

---

## 2. Labelling

The gold and validation sets. Both sheets are already labelled; these are here
for the record and for expanding the data later.

```bash
# build a sheet to label
python crawl/make_goldsheet.py                                    # gold
python crawl/make_valpool.py --dry-run                            # draw a val pool
python crawl/make_valpool.py                                      # 150 unused sentences
python crawl/make_goldsheet.py --pool data/val_pool.jsonl \
    --sheet data/val_sheet.xlsx --out data/val.jsonl              # build the val sheet

# read a finished sheet back
python crawl/make_goldsheet.py --collect                          # -> data/gold.jsonl
python crawl/make_goldsheet.py --collect --pool data/val_pool.jsonl \
    --sheet data/val_sheet.xlsx --out data/val.jsonl              # -> data/val.jsonl
```

`--pool`, `--sheet` and `--out` must be given together. Overriding only one
would write one set's rows into another set's files.

**Save and close Excel before collecting.** An open workbook holds its edits in
memory, and the collector reads the file from disk — it will silently see zero
corrections.

### Migrations, already applied

```bash
python crawl/repropose.py --dry-run                   # re-run the pre-fill rules
python crawl/repropose.py --file data/silver.jsonl    # after a rule change
python crawl/retag_english.py --dry-run               # oov -> codeswitch retag
```

Both refuse to overwrite a `human` label. Always `--dry-run` first: a
non-dry run on the wrong lexicon state once destroyed 3,928 slang expansions.

---

## 3. Training targets

```bash
python crawl/build_sft.py --dry-run --show 1   # inspect a rendered target
python crawl/build_sft.py                      # -> unsloth/data/*_reasoning.jsonl
python crawl/verify_splits.py                  # assert the four-way design holds
```

`verify_splits.py` is the integrity check: it re-derives train/dev/val/test
disjointness from the files themselves and confirms the training targets match
the silver labels. Run it after any data change.

---

## 4. Fine-tuning — **GPU**

Both models must receive the **same number of steps**, or the comparison
measures which model is faster rather than which pretraining suits Malay.

```bash
python train.py --model llama --dry-run        # config only, no GPU
python train.py --model llama --verify         # check the loss mask, ~1 min

# on a T4 (Colab): the full epoch, 1800 steps
python train.py --model llama --bs 4 --accum 1 --budget-mins 0
python train.py --model qwen  --bs 4 --accum 1 --budget-mins 0

# on a 4 GB laptop card: fit a wall-clock budget, then pin the second model
python train.py --model llama --budget-mins 120
python train.py --model qwen --max-steps <whatever llama got>
```

Useful flags: `--bs 1 --accum 8` if it runs out of memory, `--lora-attn-only`
to train 9.2M parameters instead of 24.3M, `--max-steps N` to skip the timing
probe entirely.

Each run writes `run_meta.json` beside its adapter, and the second run reads
the first one back and prints whether the two are comparable.

For Colab, upload `train_lemmatiser_t4.ipynb` and `colab_bundle.zip`.

```bash
python plot_training.py                        # loss curves -> figures/
python plot_training.py --table                # the numbers instead
```

---

## 5. Evaluation

Model **selection** happens on the validation split. The gold test set is used
**once**, for the selected model only — selecting on the test set and then
reporting its score would mean the number no longer estimates anything.

```bash
# rule baselines only, no GPU, ~30 seconds
python crawl/evaluate.py --val
python crawl/evaluate.py                       # same, on the gold test set

# neural systems — GPU, ~45-60 min each
python crawl/evaluate.py --val --base                                      # base Llama
python crawl/evaluate.py --val --base-qwen                                 # base Qwen
python crawl/evaluate.py --val --model --adapter lemmatizer_lora_llama
python crawl/evaluate.py --val --model --adapter lemmatizer_lora_qwen

# all eight systems in one table, from cache, no GPU
python crawl/evaluate.py --val --reuse \
    --adapter lemmatizer_lora_llama --adapter lemmatizer_lora_qwen

# the final number, selected model only
python crawl/evaluate.py --model --adapter lemmatizer_lora_llama --base
```

Predictions cache to `data/model_predictions_<split>_<model>.json` as each
model finishes, so a crash partway costs only the model in progress. The split
is part of the filename because it has to be: the same adapter scored on `--val`
and on gold produces two different files, and when the name carried only the
adapter they overwrote each other silently. `--reuse` rebuilds every table from
those with no GPU, which is how to regenerate report figures.

The default table is the six reported metrics — overall token accuracy,
precision, recall, F1, WER and CER — one row per language model (the
un-fine-tuned bases and the fine-tunes). `--baselines` adds the non-neural rows
back (copy, MALINDO, dict-root, rules); `--full` adds the diagnostic tables
(CHANGED / unchanged split, sentence-exact, per-band and per-source accuracy,
the honest subset, McNemar).

`--errors` prints every token where the rules and the fine-tune disagree, with
the gold answer.

---

## 6. Using it

```bash
python main.py                                 # interactive; quit / exit / q / Ctrl-C
python main.py "aku dh xleh handle keje tu"    # one sentence
python main.py --model qwen-ft "aq nk tdr"     # llama-base, qwen-base, llama-ft, qwen-ft
python main.py --stages "Budak2 tu gilaaa!!"   # show each preprocessing stage
python main.py --raw "Budak2 tu gilaaa!!"      # skip preprocessing
python main.py --plain "budak2 tu gila"        # lemma sentence only
python main.py --lang en "budak2 tu gila"      # English rule glosses
python main.py --no-lexicon "aku nk tdr"       # what the model alone produces
python main.py --no-model "aku nk tdr"         # preprocessing only, no GPU
```

In the REPL: `/raw`, `/stages`, `/plain`, `/lang`, `/help`.

### The web interface

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Same pipeline as `main.py`, with the lemma
sentence, the per-token derivation, the preprocessing stages and the raw model
output all on one page. The sidebar switches between the four evaluated models,
so a base model's format failures can be shown on demand.

The model stays in GPU memory once loaded, so switching models in the sidebar
loads a second one. On a 4 GB card, restart the app rather than flipping
between them mid-session.

### Adding a word without retraining

The model knows only what it saw in training. A word absent from the corpus is
invisible to it however well documented elsewhere.

```bash
python lexicon.py --check tdr bdh zzz          # what the lexicon makes of these
python lexicon.py --add tdo tidur              # append to data/user_lexicon.csv
python lexicon.py --list                       # everything loaded
```

`data/user_lexicon.csv` is read by two things, and it takes precedence over the
shipped `BM_dict.csv` in both:

- **preprocess.py stage 4**, which expands slang *before* the model sees it
- **lexicon.py**, which corrects the model's answer *afterwards*

The shipped dictionary does not always match this project's annotation — it maps
`nak` to `hendak` where all 38 gold occurrences say `mahu` — so the user file
records the project's own conventions. Every token where the two disagreed has
been aligned to the gold labels.

Corrections are applied **after** the model answers and are labelled `LEKSIKON`
in the output, so a reader can always tell which component produced a lemma.
Your entries always win; the shipped `BM_dict.csv` only fills gaps the model
left unresolved — it disagrees with this project's conventions in places, and a
blanket override once made a correctly-learned token wrong.

---

## 7. Inspecting the components

Each stage runs standalone, without a GPU.

```bash
python preprocess.py --demo                    # the four cleaning stages
python preprocess.py "Budak2 tu gilaaa!!"      # trace one sentence
python preprocess.py --numbers drop "bayar rm3.50"

python skill.py --card                         # the system prompt, verbatim
python skill.py --rules                        # the nine rule tags
python skill.py --demo --seed data/gold.jsonl -n 3   # rendered training targets

python crawl/try_rules.py "aq nk tdr"          # the rule baseline, CPU only
```

---

## Reference: what each split is for

| split | size | labels | used for |
|---|---|---|---|
| `unsloth/data/train_reasoning.jsonl` | 7,200 | silver | fine-tuning |
| `unsloth/data/dev_reasoning.jsonl` | 800 | silver | checkpoint selection within a run |
| `data/val.jsonl` | 150 | human | choosing between models |
| `data/gold.jsonl` | 150 | human | the reported result, used once |

The dev split cannot select between models: its labels are the rule pipeline's
own output, so a model that imitates the rules scores near-perfectly on it by
construction.
