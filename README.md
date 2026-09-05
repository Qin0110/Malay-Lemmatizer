# Malay Rojak Lemmatiser

Every command in the pipeline, in the order it runs. Anything marked **GPU**
loads a 3B model; everything else runs on CPU.

On Windows use the venv interpreter explicitly — the system Python has CPU-only
torch and fails with a confusing error. The commands below say `python` for
brevity; substitute this path:

```
E:/Malay_Lemmatizer/venv/Scripts/python.exe
```

---

## 0. Setup from a clone

The repo does not contain the model weights: a LoRA adapter is 110–126 MB and
GitHub rejects any file over 100 MB. Either train them (section 4) or extract
the two adapter folders into `unsloth/`:

```
unsloth/lemmatizer_lora_llama/
unsloth/lemmatizer_lora_qwen/
```

Check that `run_meta.json` in each reports the same step count, or the
comparison between them measures schedule rather than pretraining.

Without the weights everything except generation still works.

---

## 1. Data collection — already done, do not re-run

Re-running these produces different sentence ids and orphans the 300
hand-labelled sentences, which cannot be recovered.

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

Both sheets are already labelled; these are here for expanding the data later.

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

`--pool`, `--sheet` and `--out` must be given together, or one set's rows land
in another set's files.

**Save and close Excel before collecting.** The collector reads from disk, so an
open workbook silently yields zero corrections.

### Migrations, already applied

```bash
python crawl/repropose.py --dry-run                   # re-run the pre-fill rules
python crawl/repropose.py --file data/silver.jsonl    # after a rule change
python crawl/retag_english.py --dry-run               # oov -> codeswitch retag
```

Both refuse to overwrite a `human` label. Always `--dry-run` first: a non-dry
run on the wrong lexicon state once destroyed 3,928 slang expansions.

---

## 3. Training targets

```bash
python crawl/build_sft.py --dry-run --show 1   # inspect a rendered target
python crawl/build_sft.py                      # -> unsloth/data/*_reasoning.jsonl
python crawl/verify_splits.py                  # assert the four-way design holds
```

Run `verify_splits.py` after any data change: it re-derives train/dev/val/test
disjointness from the files and confirms the targets match the silver labels.

---

## 4. Fine-tuning — **GPU**

QLoRA: the 4-bit base is frozen, LoRA adapters (r=16, α=16) train on all seven
projections. Both models must receive the **same number of steps**, or the
comparison measures which model is faster rather than which pretraining suits
Malay.

```bash
python train.py --model llama --dry-run        # config only, no GPU
python train.py --model llama --verify         # check the loss mask, ~1 min

# the full epoch, 1800 steps, at the default batch 4 x 1
python train.py --model llama
python train.py --model qwen
```

Useful flags: `--bs 1 --accum 8` if it runs out of memory, `--lora-attn-only`
to train 9.2M parameters instead of 24.3M, `--max-steps N` to stop early.

Each run writes `run_meta.json` beside its adapter, and the second run reads the
first one back and prints whether the two are comparable.

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
# generate predictions — GPU, ~45-60 min each
python crawl/evaluate.py --val --base                                      # base Llama
python crawl/evaluate.py --val --base-qwen                                 # base Qwen
python crawl/evaluate.py --val --model --adapter lemmatizer_lora_llama
python crawl/evaluate.py --val --model --adapter lemmatizer_lora_qwen

# every cached model in one table, no GPU
python crawl/evaluate.py --val --reuse \
    --adapter lemmatizer_lora_llama --adapter lemmatizer_lora_qwen

# the final number, selected model only
python crawl/evaluate.py --reuse --adapter lemmatizer_lora_llama
```

Predictions cache to `data/model_predictions_<split>_<model>.json` as each model
finishes, so a crash partway costs only the model in progress. The split is part
of the filename because the same adapter scored on `--val` and on gold produces
two different files. `--reuse` rebuilds the tables from cache with no GPU, and
scores every base and adapter that has one.

Two tables print: the six reported metrics (overall token accuracy, precision,
recall, F1, WER, CER), then the confusion matrix behind them. The matrix cells
sum to slightly more than the token count — a token that needed a change and
received the wrong one counts as both a false positive and a false negative.

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
output on one page. It serves the selected fine-tune only; use `main.py --model`
to run another. The model stays in GPU memory once loaded — on a 4 GB card,
restart the app rather than loading a second one.

### Adding a word without retraining

The model knows only what it saw in training.

```bash
python lexicon.py --check tdr bdh zzz          # what the lexicon makes of these
python lexicon.py --add tdo tidur              # append to data/user_lexicon.csv
python lexicon.py --list                       # everything loaded
```

`data/user_lexicon.csv` takes precedence over the shipped `BM_dict.csv` in both
places it is read:

- **preprocess.py stage 4**, which expands slang *before* the model sees it
- **lexicon.py**, which corrects the model's answer *afterwards*

The shipped dictionary does not always match this project's annotation — it maps
`nak` to `hendak` where all 38 gold occurrences say `mahu` — so the user file
records the project's own conventions.

Corrections are applied after the model answers and are labelled `LEKSIKON`, so
a reader can tell which component produced a lemma. Your entries always win; the
shipped `BM_dict.csv` only fills gaps the model left unresolved, because a
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
