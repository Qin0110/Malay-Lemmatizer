# Malay Rojak Lemmatiser

A lemmatiser for informal Malaysian Malay ("bahasa rojak") - the kind of text
people write in comment sections and chats, full of slang, short forms, English
words, names and creative spelling.

```
input    Aku dh xleh handle keje tu, Kak Uji pun tau
lemmas   aku sudah tidak boleh handle kerja itu kakak Uji pun tau
```

Every word comes back with the rule that produced its lemma:

| token | lemma | rule |
| --- | --- | --- |
| dh | sudah | RINGKAS - slang expanded |
| xleh | tidak boleh | RINGKAS |
| keje | kerja | RINGKAS |
| Uji | Uji | NAMA - proper noun, kept as written |
| pun | pun | PARTIKEL - discourse particle, kept |
| handle | handle | INGGERIS - English, kept |

There are two lemmatisers, and they work together:

1. **Rules** - a slang lexicon, the MALINDO Morph dictionary and spelling
   repair (elongation, `2` reduplication, `x`-negation). Runs on any computer.
2. **A fine-tuned LLM** - Llama 3.2 3B trained with QLoRA on 8,820 sentences
   the rules labelled. Needs an NVIDIA GPU, and is optional. The trained
   adapter is a separate download, which `SETUP.bat` fetches for you.

When both run, every word the rules are sure about keeps the rule's label - the
same label it had in the training data - and the model decides the rest.

---

## Contents

- [Results](#results)
- [Quick start (Windows)](#quick-start-windows)
- [Manual install (any OS)](#manual-install-any-os)
- [Using the app](#using-the-app)
- [Command-line tools](#command-line-tools)
- [Retraining (optional)](#retraining-optional)
- [Adding more data](#adding-more-data)
- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Data and privacy](#data-and-privacy)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)
- [Credits and licences](#credits-and-licences)

---

## Results

Two hand-checked sets of 150 sentences each: **val** (1,868 tokens), used to
choose between systems, and **gold** (1,786 tokens), the final test.

| System | Val accuracy | Val F1 | Gold accuracy | Gold F1 | Gold, hand-corrected words |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rules | 91.8% | 84.6% | 96.0% | 93.3% | 47.7% |
| Llama 3.2 3B + LoRA | 90.9% | 81.9% | 95.0% | 90.8% | 47.7% |
| **Llama + LoRA + rules** (the app's default) | 91.5% | 83.4% | 95.6% | 92.2% | **50.0%** |
| Earlier adapter (7,200 sentences, no NAMA rule) | 84.3% | 80.4% | - | - | - |
| Llama 3.2 3B, not fine-tuned | 18.8% | 2.6% | - | - | - |

- **Precision, recall and F1** treat lemmatisation as the decision to change a
  word, so copying the input scores 0. About two thirds of the words are already
  their own lemma, which is why accuracy is always much higher than F1.
- **Hand-corrected words** are the tokens a person changed while labelling. The
  rest of each label sheet started as the rules' output, so the rules get those
  for free - this column is the fairest comparison, and the combined system wins
  it.
- The released adapter never broke its answer format on either set.
- Against the earlier adapter it gains 6.6 points of accuracy and 1.5 of F1 on
  val, from more training data, case-aware input and the NAMA rule.

Reproduce the rule rows (no GPU):

```bash
python crawl/evaluate.py --val
```

The model rows (GPU, about 30 minutes per set):

```bash
python crawl/evaluate.py --val --model llama-ft
```

```bash
python crawl/evaluate.py --model llama-ft
```

---

## Quick start (Windows)

1. **Double-click `SETUP.bat`.** It creates a virtual environment, installs the
   packages and checks that everything works. A few minutes, needs internet.

   If an NVIDIA GPU is found, it asks whether to also install the fine-tuned
   model: PyTorch, Unsloth, the Llama 3.2 3B base model and the trained
   adapter, about 7 GB. Answer **N** if you only want the rules - you can
   add it later with `SETUP.bat gpu`.
2. **Double-click `RUN.bat`.** The app opens in your browser at
   <http://localhost:8501>. Close the console window to stop it.

After the first setup, `RUN.bat` is all you need.

Setup uses an installed Python 3.10-3.12 if it finds one, otherwise it
downloads a private copy. Installing Python 3.12 from
[python.org](https://www.python.org/downloads/) first is recommended (see
[Troubleshooting](#troubleshooting)).

Nothing is installed system-wide and no admin rights are needed. Everything
goes into one folder, and deleting it uninstalls everything:

```
%LOCALAPPDATA%\RojakLemma\rt\     private Python (only if none was found)
%LOCALAPPDATA%\RojakLemma\venv\   the virtual environment
%LOCALAPPDATA%\RojakLemma\hf\     downloaded base models
```

`HOW_TO_RUN.txt` has the same steps in more detail.

---

## Manual install (any OS)

Python 3.10-3.12.

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

```bash
pip install -r requirements-app.txt
```

For the fine-tuned model (NVIDIA GPU, driver 528 or newer):

```bash
pip install -r requirements-gpu.txt
```

```bash
python download_model.py
```

The second command downloads the trained adapter (about 100 MB) from the
GitHub release into `models/`. The base model (2.2 GB) is downloaded the
first time the model runs.

Check the install, then start the app:

```bash
python verify_setup.py
```

```bash
streamlit run app.py
```

On Linux, `pyenchant` needs the Enchant library:
`sudo apt install libenchant-2-2 hunspell-en-us`. Without it English words are
tagged KEKAL instead of INGGERIS; the lemmas are the same.

---

## Using the app

Type one or more sentences and press **Lemmatise** (or Ctrl+Enter).

You get the lemma sentence and a table with one row per word:

| Column | Meaning |
| --- | --- |
| token | the word as written |
| lemma | its lemma |
| rule | the rule behind it; `?` means the rules were not sure |
| meaning | what the rule does |
| decided by | `rules` or `model` |
| model said | the model's answer, when it was replaced by the training-data label |

The sidebar has:

| Setting | Effect |
| --- | --- |
| Lemmatiser | `Rules only`, the fine-tuned model `llama-ft`, or the untrained base models for comparison. Models are listed once the GPU packages are installed |
| Follow the training rules exactly | On (default): slang, particles, names, numbers, elongation and reduplication always get the label they had in the training data. Off: the model's own answer for every word |
| Rule glosses | Malay or English descriptions |

The first sentence with a model takes 30-60 seconds while it loads.

### The rules

| Rule | What happened | Example |
| --- | --- | --- |
| RINGKAS | slang or short form expanded | dh -> sudah, xleh -> tidak boleh |
| IMBUHAN | affix removed | berjalan -> jalan |
| GANDA | reduplication resolved | budak2 -> budak |
| PANJANG | elongation collapsed | gilaaa -> gila |
| PARTIKEL | discourse particle, kept | la, weh, kot, je |
| NAMA | proper noun, kept as written | Safwan, Malaysia, Kak Uji |
| INGGERIS | English word, kept | handle, best |
| NOMBOR | number | 2, 2026 |
| KAMUS | already a dictionary lemma | makan |
| KEKAL | not recognised, left as it is | |

---

## Command-line tools

Run these from the project folder with the environment active
(`%LOCALAPPDATA%\RojakLemma\venv\Scripts\activate` if you used `SETUP.bat`).

```bash
python main.py "aku dh xleh handle keje tu"
```

Lemmatise in the terminal. Without a sentence it starts an interactive prompt.
`--rules-only` skips the model, `--no-follow-rules` shows the model's own
answer for every word, `--stages` shows the cleaning.

When you run the model yourself rather than through `RUN.bat`, point it at the
base model `SETUP.bat` already downloaded, or it is downloaded again:

```bat
set HF_HOME=%LOCALAPPDATA%\RojakLemma\hf
```

```bash
python crawl/try_rules.py "aq nk tdr dlu, esok kite jumpe kat umah"
```

The rules alone, with the tag for every word and what the dictionary alone
would give.

```bash
python lexicon.py --check xleh nak hg
```

Where a slang word's expansion comes from. `--add tdo tidur` adds your own
entry to `data/user_lexicon.csv`; your entries take priority over the built-in
list and `BM_dict.csv`.

```bash
python lexicon.py --add-name "Berjaya Grocer"
```

Always keep a name as written, even when it is typed in lowercase or is made of
ordinary words. Names go in `data/user_names.txt`, one per line; it already
holds common Malaysian places. The app picks both files up straight away -
no retraining needed. Rebuild the training data (`crawl/build_silver.py`,
`crawl/build_sft.py`) if you later retrain.

```bash
python preprocess.py --demo
```

How input is cleaned before it is lemmatised.

---

## Retraining (optional)

You do not need this to use the lemmatiser - the trained adapter is a
download. Retrain if you change the data, the lexicon or the rules. Each step
checks the previous one, so run them in this order:

```bash
python crawl/build_silver.py
```

Labels every corpus sentence with the rules and splits them 90/10 into train
and dev. Sentences in the val and gold sets are never used.

```bash
python crawl/build_sft.py
```

Writes the chat-format training files to `data/sft/`. It refuses to run if the
labels are out of date with the rules, so the model is never trained on
labels the app would contradict.

```bash
python crawl/verify_splits.py
```

Confirms train, dev, val and gold do not overlap and every target matches its
labels.

```bash
python train.py --model llama
```

Fine-tunes Llama 3.2 3B (one epoch, about 2,200 steps). The adapter is saved to
`models/lemmatizer_lora_llama/`, replacing the downloaded one. Use
`--model qwen` for Qwen2.5 3B. On a 4 GB GPU use `--bs 1 --accum 4` - that
takes many hours on a laptop. `--dry-run` shows the settings without a GPU
and `--verify` checks the training setup in a minute before a long run.

```bash
python crawl/evaluate.py --val --model llama-ft --model qwen-ft
```

Scores the rules, each model, and each model combined with the rules, on the
validation set. Model outputs are saved, so `--reuse` re-scores them without a
GPU. Pick the best system on val, then report it once on gold:

```bash
python crawl/evaluate.py --model llama-ft
```

`python plot_training.py` draws the loss curves into `figures/`.

---

## Adding more data

New sentences go through the same path whatever the source:

```
posts -> data/raw/*.jsonl -> crawl/build_corpus.py -> data/corpus.jsonl -> build_silver -> build_sft -> train
```

`build_corpus.py` only ever appends. Existing sentences keep their ids, so the
hand-labelled val and gold sets stay valid.

### YouTube comments

Needs a free YouTube Data API v3 key in a `.env` file in the project folder:
`YOUTUBE_API_KEY=...`

```bash
python crawl/youtube_crawl.py --probe
```

```bash
python crawl/youtube_crawl.py --videos 25
```

`--channel @handle` crawls other channels; `--find "channel name"` looks up
handles. Re-running resumes where it stopped.

### Twitter/X, Threads, Facebook, Instagram, Reddit, forums

Export or collect the posts yourself, then import the file:

```bash
python crawl/import_posts.py tweets.js --source twitter
```

```bash
python crawl/import_posts.py threads.json --source threads
```

```bash
python crawl/import_posts.py comments.csv --source facebook --field message
```

It reads `.txt` (one post per line), `.csv`, `.json`, `.jsonl` and the X archive's
`tweets.js`, and keeps only the text. Your own account export ("Download an
archive of your data" on X, "Download your information" on Meta) is the easiest
source. Check a platform's terms before collecting other people's posts - X and
Reddit in particular restrict using their content to train models.

Then add the new sentences to the corpus and rebuild:

```bash
python crawl/build_corpus.py
```

`--dry-run` shows what would be added; `--validate` writes a sample to
hand-check, and `python crawl/score_filter.py` turns it into precision and
recall for the rojak filter.

### More hand-labelled data

```bash
python crawl/make_valpool.py --out data/test2_pool.jsonl
```

Draws 150 new sentences (held out of training from then on), and prints the
commands to build an Excel labelling sheet from them and read it back.

---

## How it works

### One cleaning step for training and use

The model is trained on sentences cleaned by `preprocess.py` - links, @handles,
emails, phone numbers, emoji and symbols removed, case and spelling left alone.
The app and `main.py` prepare input with the same function, so the model never
sees text in a form it was not trained on. (An earlier version lowercased the
input and expanded slang from a different dictionary before the model saw it,
which is how answers drifted from the training labels.)

### The rule cascade

`rules.py` labels each word, in this order:

1. **NAMA** - proper nouns, judged on the text as written:
   - anything listed in `data/user_names.txt`, in any case
   - two or more capitalised words in a row (Berjaya Grocer, Mydin Mall,
     Rapid KL), unless the whole sentence is in Title Case
   - a capitalised word that MALINDO only lists with a capital (Malaysia,
     Allah), that the corpus almost always capitalises, that follows a title
     (Kak, Pak, Dato, Tun), or that is neither a Malay nor an English word
2. numbers and discourse particles
3. the slang lexicon: `data/user_lexicon.csv`, then the built-in list in
   `lexicon.py`, then `malindo_morph/BM_dict.csv`. `pgi`/`pg` is read from its
   neighbours: *pergi* after `nak`, `jom`, `tak`... or before `ke`, `kat` or a
   name; *pagi* after a number or `esok`, `setiap`, `pukul`...
4. `2` reduplication, then elongation (`gilaaa` -> `gila`)
5. the MALINDO root column, which removes affixes
6. the MALINDO lemma column
7. the English dictionary, otherwise left unchanged

Lemmas are lowercase except proper nouns, which keep their spelling.

### Rules and model together

Rules marked as certain (slang, particles, names, numbers, reduplication,
elongation, dictionary words, English) are exactly what the training data says.
With "Follow the training rules exactly" on, those words keep the rule label and
the model's answer is used for the uncertain ones - ambiguous words, affixes and
unknown spellings - where it can use the context of the sentence.

### The fine-tune

| | |
| --- | --- |
| Base | `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` or `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` |
| Method | QLoRA with Unsloth, loss on the answer only |
| LoRA | r = 16, alpha = 16, all attention and MLP projections |
| Optimiser | adamw_8bit, lr 2e-4, linear schedule, 3% warm-up |
| Batch | 4 (1 x 4 gradient accumulation on a 4 GB GPU) |
| Max length | 1,024 tokens |
| Released adapter | step 1,323 of 2,205 (60% of an epoch), the lowest dev loss (0.0183) |

The model answers with one line per word (`token | RULE | lemma`) and the full
lemma sentence, so every answer can be checked word by word. The prompt is in
`skill.py`; each adapter records which prompt version it was trained with.

---

## Project structure

```
SETUP.bat / RUN.bat        Windows installer and launcher
HOW_TO_RUN.txt             Step-by-step guide for Windows
app.py                     Streamlit app
main.py                    Command-line lemmatiser
pipeline.py                Rules + model, shared by the app, main.py and evaluate.py
rules.py                   The rule-based lemmatiser (also writes the training labels)
lexicon.py                 Slang lexicon and particles
preprocess.py              Text cleaning
skill.py                   Prompts, rule tags, answer parsing
train.py                   QLoRA fine-tuning
plot_training.py           Loss curves
verify_setup.py            Installation check
download_model.py          Fetches the trained adapter from the GitHub release

crawl/
  youtube_crawl.py         Collect YouTube comments
  import_posts.py          Import exported social media posts
  build_corpus.py          Filter new posts into the corpus
  score_filter.py          Precision/recall of the rojak filter
  build_silver.py          Rule-label the corpus, train/dev split
  build_sft.py             Training files for the model
  verify_splits.py         Overlap and consistency checks
  make_valpool.py          Draw sentences for hand labelling
  make_goldsheet.py        Build / collect an Excel labelling sheet
  evaluate.py              Score every system
  try_rules.py             The rules on your own sentence

data/
  corpus.jsonl             10,099 filtered rojak sentences
  silver.jsonl             The corpus labelled by the rules (9,799)
  split_manifest.json      Which sentences are train and dev
  sft/                     Training files (8,820 train, 979 dev)
  val.jsonl                150 hand-checked sentences, for choosing models
  gold.jsonl               150 hand-checked sentences, the final test
  *_pool.jsonl             The sentences each labelled set was drawn from
  *_sheet.xlsx             The labelling sheets
  filter_check.xlsx        Hand-checked sample of the rojak filter
  user_lexicon.csv         Your slang entries
  user_names.txt           Names to always keep as written

malindo_morph/
  malindo_dic_2024.tsv     MALINDO Morph dictionary
  BM_dict.csv              Rojak -> standard spellings

models/                    Fine-tuned adapters (downloaded, not in git)
```

---

## Data and privacy

The sentences in `data/` come from public YouTube comments. Only the comment
text is kept:

- no author names, channel handles, profile links, comment ids or video ids
- links, @mentions, emails, phone numbers and timestamps are removed

Raw crawls and imports (`data/raw/`) stay on your machine and are not part of
the repository. If you recognise a comment of yours and want it removed, open an
issue.

---

## Troubleshooting

| Problem | Fix |
| --- | --- |
| Anything goes wrong on Windows | Run `SETUP.bat` again. It is safe to re-run and ends with a full check |
| `DLL load failed ... Application Control policy has blocked this file` | Windows Smart App Control blocked the unsigned Python that setup downloaded. Install Python 3.12 from [python.org](https://www.python.org/downloads/), delete `%LOCALAPPDATA%\RojakLemma` and run `SETUP.bat` again |
| Only "Rules only" in the Lemmatiser list | The GPU packages are not installed: run `SETUP.bat gpu` |
| No `llama-ft` in the list | The adapter is not in `models/`: run `python download_model.py` |
| "CUDA is not available" | Update the NVIDIA driver (528 or newer) |
| Out of GPU memory | A 3B model needs about 4 GB. Close other GPU programs; for training use `--bs 1 --accum 4` |
| `build_sft.py` says the labels are out of date | You changed the lexicon or rules. Run `python crawl/build_silver.py` first |
| Port 8501 already in use | The app is already running in another window |
| English words tagged KEKAL | `pyenchant` is missing. Optional |

---

## Limitations

- Built for Malaysian Malay as written in YouTube comments. Other registers,
  Indonesian or mostly-English text will do worse.
- The training labels come from the rules, so the model learns their mistakes
  too. On the tokens a person had to correct in val, the rules get only 36%
  right - more hand-labelled data is the most useful next step.
- Proper nouns are mostly detected from capitalisation. A name typed in
  lowercase is treated as an ordinary word unless it is in
  `data/user_names.txt`, and an ordinary word capitalised next to a name can
  occasionally be kept as written.
- The test sets are small (150 sentences each), so differences of a point or two
  are within noise.

---

## Credits and licences

The code is released under the MIT Licence - see [LICENSE](LICENSE).

**MALINDO Morph** (`malindo_morph/malindo_dic_2024.tsv`) is licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/):

> Nomoto, Hiroki, Hannah Choi, David Moeljadi and Francis Bond. 2018.
> MALINDO Morph: Morphological dictionary and analyser for Malay/Indonesian.
> *Proceedings of the LREC 2018 Workshop "The 13th Workshop on Asian Language
> Resources"*, 36-43. <https://github.com/matbahasa/MALINDO_Morph>

**Base models.** Fine-tuned adapters are derived from
[Llama 3.2 3B Instruct](https://www.llama.com/llama3_2/license/) (Llama 3.2
Community License - Built with Llama) or
[Qwen2.5 3B Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE)
(Qwen Research License, non-commercial). Training uses [Unsloth](https://github.com/unslothai/unsloth).
