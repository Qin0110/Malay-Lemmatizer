import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# region: Uncomment this when run on Windows to avoid DLL load failed error
os.add_dll_directory(r"C:\vcpkg\installed\x64-windows\bin")

import logging
logging.basicConfig(level=logging.INFO)

import malaya
import pandas as pd
import re

from malaya.dictionary import is_english as _malaya_is_english
from malaya.dictionary.english.words import words as _english_words

# e.g. "makan2" -> "makan-makan", keeps trailing punctuation like "budak2,"
REDUP_RE = re.compile(r'^([A-Za-z]{2,})2([^A-Za-z0-9]*)$')

# malaya.dictionary builds its English set as
#     ENGLISH_WORDS = {i for i in _english_words if len(i) > 2} | {'me','on','of'}
# so every English word of two letters or fewer is invisible to is_english(),
# except the three added back by hand. That is what lets malaya's spelling
# corrector treat "my" as vowel-less Malay shorthand and run its vowel
# augmentation on it ("my" -> "may"/"muy"/"miy"/"mey"), picking "muy" because
# it happens to be in the Malay word list. Same mechanism gives "is" -> "isu".
#
# The words below are the short forms that actually turn up in rojak
# code-switching, restored from malaya's own English list. Deliberately NOT the
# full 127 dropped entries: that set includes "di", "ke", "ha", "ni", "ma" and
# other strings that are real Malay words or genuine rojak shorthand, and
# shielding those would block the corrections this normalizer exists to make.
SHORT_ENGLISH = {
    'my', 'is', 'it', 'we', 'he', 'be', 'do', 'am', 'an', 'as', 'at',
    'by', 'if', 'in', 'no', 'or', 'so', 'to', 'up', 'us', 'ok',
}

# Keep only forms malaya's own English word list actually contains, so a typo
# here cannot silently shield a non-word.
SHORT_ENGLISH = {w for w in SHORT_ENGLISH if w in set(_english_words)}


def is_english(word):
    """malaya's is_english plus the short words its len>2 filter discards.

    Passed to normalize(check_english_func=...) so the corrector's existing
    English guard fires on "my" and "is" instead of rewriting them.
    """
    return word.lower() in SHORT_ENGLISH or _malaya_is_english(word)


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

        return self.normalizer.normalize(
            string, check_english_func=is_english
        )['normalize']
