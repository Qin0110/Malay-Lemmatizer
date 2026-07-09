import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.add_dll_directory(r"C:\vcpkg\installed\x64-windows\bin")

import logging
logging.basicConfig(level=logging.INFO)

import malaya

class Normalizer:
    def __init__(self):
        self.lm = malaya.language_model.kenlm(model = 'bahasa-wiki-news')
        self.corrector = malaya.spelling_correction.probability.load(language_model = self.lm)
        self.stemmer = malaya.stem.huggingface()
        self.normalizer = malaya.normalizer.rules.load(self.corrector, self.stemmer)

    def normalize(self, string):
        return self.normalizer.normalize(string)['normalize']