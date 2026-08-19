import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# region: Uncomment this when run on Windows to avoid DLL load failed error
os.add_dll_directory(r"C:\vcpkg\installed\x64-windows\bin")

import logging
logging.basicConfig(level=logging.INFO)

import malaya
import pandas as pd
import re

# e.g. "makan2" -> "makan-makan", keeps trailing punctuation like "budak2,"
REDUP_RE = re.compile(r'^([A-Za-z]{2,})2([^A-Za-z0-9]*)$')

class Normalizer:
    def __init__(self):
        self.slang_dict = pd.read_csv('malindo_morph/BM_dict.csv')
        self.lm = malaya.language_model.kenlm(model = 'bahasa-wiki-news')
        self.corrector = malaya.spelling_correction.probability.load(language_model = self.lm)
        self.stemmer = malaya.stem.huggingface()
        self.normalizer = malaya.normalizer.rules.load(self.corrector, self.stemmer)

    def normalize(self, string):
        tokens = string.split()
        for i, tok in enumerate(tokens):
            redup = REDUP_RE.match(tok)
            if redup:
                base, tail = redup.group(1), redup.group(2)
                tok = tokens[i] = '{0}-{0}{1}'.format(base, tail)

            match = self.slang_dict.loc[self.slang_dict['rojak'] == tok.lower(), 'actual']
            if not match.empty:
                tokens[i] = match.iloc[0]
        string = ' '.join(tokens)

        return self.normalizer.normalize(string)['normalize']