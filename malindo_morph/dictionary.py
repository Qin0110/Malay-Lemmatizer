import os
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

    def surface_lexicon(self):
        """Every surface form MALINDO knows, lowercased.

        The language identifier needs set membership, not a row lookup, to
        tell a Malay token from an English one.
        """
        if self._lexicon is None:
            self._lexicon = set(self.df["surface_lower"].dropna())
        return self._lexicon

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
        # Reduplication is the exception. "root" collapses "makan-makan" to
        # "makan", losing the reduplication the surface form carries, so for
        # rows MALINDO marks as reduplicated keep the headword instead. field()
        # already takes the first alternant of "gunting@menggunting".
        if field("reduplication") not in ("", "0"):
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