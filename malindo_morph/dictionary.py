import pandas as pd
from collections import defaultdict
from pathlib import Path


class Dictionary:
    def __init__(self, dictionary_path=None):
        if dictionary_path is None:
            current_dir = Path(__file__).parent
            self.dictionary_path = current_dir / "malindo_dic_2024.tsv"
        else:
            self.dictionary_path = Path(dictionary_path)

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

        self.df = self._load_tsv()
        self.surface_index = self._build_surface_index()

    def _load_tsv(self):
        if not self.dictionary_path.exists():
            raise FileNotFoundError(f"File not found: {self.dictionary_path}")

        df = pd.read_csv(
            self.dictionary_path,
            sep="\t",
            names=self.column,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8",
            on_bad_lines="skip",
        )
        
        # df = df[~df["id"].str.startswith("ex-", na=False)] (di-- prefix wouldn't be found in dictionary)
        
        # Normalize text for easier lookup
        df["surface_lower"] = df["surface"].str.lower()
        return df

    def _build_surface_index(self):
        index = defaultdict(list)

        for row in self.df.to_dict("records"):
            surface = row["surface_lower"]
            index[surface].append(row)

        return index

    def lookup(self, word: str):
        word = word.lower().strip()
        return self.surface_index.get(word, [])

    def lookup_first(self, word: str):
        results = self.lookup(word)

        if not results:
            return None

        return results[0]