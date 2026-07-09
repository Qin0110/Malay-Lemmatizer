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
    
    def morphological_analysis(self, word):
        grammar = self.df.loc[self.df["surface"] == word.lower()]
        return ({
            "ID": grammar["id"].to_string(index=False),
            "Word": grammar["surface"].to_string(index=False),
            "Root": grammar["root"].to_string(index=False),
            "Prefix": grammar["prefix"].to_string(index=False),
            "Suffix": grammar["suffix"].to_string(index=False),
            "circumfix": grammar["circumfix"].to_string(index=False),
            "Lemma": grammar["lemma"].to_string(index=False),
        })