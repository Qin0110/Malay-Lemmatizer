import os
from collections import Counter

import pandas as pd

from Preprocess.tokenizer import Tokenizer

class Dictionary:
    def __init__(self, dictionary_path=None):
        self.tokenizer = Tokenizer()
        self.dictionary_path = "malindo_morph/malindo_dic_2024.tsv"
        #region Column DEFINE by MALINDO
        #github source: https://github.com/matbahasa/MALINDO_Morph/tree/master

        self.column = [
            "id",
            "root",          # Akar
            "surface",       # Bentuk lahir
            "prefix",        # Awalan/proklitik
            "suffix",        # Akhiran/enklitik
            "circumfix",     # Apitan
            "reduplication", # Penggandaan
            "source",        # Sumber
            "stem",          # Dasar
            "lemma",         # Lema
        ]

        self._lexicon = None
        self._redup_index = None
        self.ganda_semu = self.load_ganda_semu()
        self.df = self.load()

    def load(self):
        if not os.path.exists(os.path.join(os.getcwd(), self.dictionary_path)):
            raise FileNotFoundError(f"File not found: {self.dictionary_path}")

        df = pd.read_csv(
            self.dictionary_path, 
            sep = "\t",
            names = self.column,
            encoding = "utf-8"
        )
        
        # df = df[~df["id"].str.startswith("ex-", na=False)] (di-- prefix wouldn't be found in dictionary)
        
        # Normalize text for easier lookup
        df["surface_lower"] = df["surface"].str.lower()
        return df

    GANDA_SEMU_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "ganda_semu.txt")

    def load_ganda_semu(self, path=None):
        """Reduplications that are themselves the lemma, one per line.

        MALINDO cannot supply this. It files "rama" under the headword "rama"
        exactly as it files "budak" under "budak", so no column separates
        "rama-rama" (a butterfly, never a repeated "rama") from "budak-budak"
        (simply more than one "budak"). The distinction is semantic, so the
        list is curated; ganda_review.tsv holds the borderline calls.

        A missing file degrades to resolving everything, which is what this
        code did before the list existed, rather than failing the load.
        """
        try:
            with open(path or self.GANDA_SEMU_PATH, encoding="utf-8") as fh:
                return {line.strip().lower() for line in fh
                        if line.strip() and not line.startswith("#")}
        except OSError:
            return set()

    def surface_lexicon(self):
        """Every surface form MALINDO knows, lowercased.

        The language identifier needs set membership, not a row lookup, to
        tell a Malay token from an English one.
        """
        if self._lexicon is None:
            self._lexicon = set(self.df["surface_lower"].dropna())
        return self._lexicon

    def reduplication_index(self):
        """Map a reduplication's first element to the full forms MALINDO attests.

        The "budak2" shorthand writes only the first element and leaves the
        reader to supply the second, but which second element is correct is
        lexical, not mechanical. "berjalan2" is "berjalan-jalan" -- the prefix
        is not repeated -- while "pelajar2" is "pelajar-pelajar", where it is;
        and "berita2" is "berita-berita" because "ber-" there is not a prefix
        at all. Rather than hand-code that split and get those three classes
        wrong, ask MALINDO what it actually records for each first element.

        Keyed by (first element, trailing enclitic) so "orang2" and "orang2an"
        resolve to "orang-orang" and "orang-orangan" separately. Values count
        rows per surface form: MALINDO carries one dictionary row plus one row
        per corpus attestation, so the count is a weak frequency signal the
        caller uses to rank competing expansions.
        """
        if self._redup_index is not None:
            return self._redup_index

        def value(cell):
            if pd.isna(cell):
                return ""
            cell = str(cell).strip()
            return "" if cell == "0" else cell

        index = {}
        for surface, suffix, circumfix, redup in zip(
            self.df["surface_lower"],
            self.df["suffix"],
            self.df["circumfix"],
            self.df["reduplication"],
        ):
            # Only rows MALINDO marks as reduplicated, so hyphenated compounds
            # ("mata-hari") and names cannot pose as expansions.
            if not value(redup) or pd.isna(surface):
                continue
            parts = str(surface).split("-")
            if len(parts) != 2 or not all(parts):
                continue

            # The enclitic rides in whichever column MALINDO used: "-an" lands
            # in suffix, but "se--nya" and "ke--an" put it after the "--".
            suffix, circumfix = value(suffix), value(circumfix)
            if suffix:
                tail = suffix.lstrip("-")
            elif "--" in circumfix:
                tail = circumfix.split("--", 1)[1]
            else:
                tail = ""

            index.setdefault((parts[0], tail), Counter())[str(surface)] += 1

        self._redup_index = index
        return index

    def morphological_analysis(self, word):
        grammar = self.df.loc[self.df["surface_lower"] == word.lower()]

        # Out-of-vocabulary: keep the original token instead of pandas' empty-Series repr.
        if grammar.empty:
            return {
                "ID": "",
                "Word": word,
                "Root": "",
                "Prefix": "",
                "Suffix": "",
                "circumfix": "",
                "Lemma": word,
                "Candidates": [],
            }

        # A surface form can map to several entries, including one row per
        # casing variant ("ORG", "Org", "org") whose order is arbitrary. Prefer
        # the row that matches the token's own casing so a lowercase token does
        # not come back shouting; otherwise fall back to the first row.
        exact = grammar.loc[grammar["surface"].astype(str) == word]
        first = (exact if not exact.empty else grammar).iloc[0]

        def field(name):
            value = first[name]
            if pd.isna(value):
                return ""
            # MALINDO encodes alternations as "makan@memakan"; keep the first.
            return str(value).split("@")[0].strip()

        # "anak+dia" / "satu+orang" append a clitic or affix marker to the head
        # word; "root" holds that head, so prefer it over the raw notation.
        lemma = field("lemma")
        if "+" in lemma:
            lemma = field("root") or lemma.split("+")[0].strip()

        # MALINDO's "lemma" column is the dictionary headword, which is not the
        # base form this lemmatizer wants:
        #   - unaffixed roots are filed under a derived verb ("sampai" ->
        #     "menyampai", "makan" -> "memakan"), which would make a lemmatizer
        #     *add* affixes;
        #   - affixed forms just repeat themselves ("memakan" -> "memakan").
        # The "root" column holds the actual base in both cases, so prefer it
        # and fall back to the headword only when root is missing.
        #
        # Reduplication is the exception, and it splits two ways.
        #
        # A full X-X doubling is usually plural or intensive, and "root" is the
        # lemma: budak-budak -> budak, lagu-lagu -> lagu. Kata ganda semu is
        # not -- "rama-rama" is a butterfly and "rama" is not a small one --
        # so those keep the doubled form. MALINDO's headword cannot make the
        # call (it lists "budak-budak" as an entry too), so ganda_semu decides.
        #
        # Partial and rhythmic reduplication ("tukar-menukar", "muda-mudi",
        # "berjam-jam") is left on the headword as before: the two halves
        # differ, so there is no base to collapse onto.
        if field("reduplication") not in ("", "0"):
            halves = word.lower().split("-")
            if len(halves) == 2 and halves[0] == halves[1]:
                lemma = (word.lower() if word.lower() in self.ganda_semu
                         else field("root") or lemma)
            else:
                lemma = field("lemma") or field("root") or lemma
        else:
            lemma = field("root") or lemma

        return {
            "ID": field("id"),
            "Word": field("surface"),
            "Root": field("root"),
            "Prefix": field("prefix"),
            "Suffix": field("suffix"),
            "circumfix": field("circumfix"),
            "Lemma": lemma or word,
            "Candidates": sorted({
                str(v).split("@")[0].strip()
                for v in grammar["lemma"].dropna().tolist()
            }),
        }