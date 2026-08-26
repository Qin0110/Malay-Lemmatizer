import re

from malindo_morph.dictionary import Dictionary
from Preprocess.normalizer import Normalizer
from Preprocess.tokenizer import Tokenizer

string = 'boleh dtg 8pagi esok tak atau minggu depan? 2 oktober 2019 2pm, tlong bayar rm 3.2k sekali tau'

normalizer = Normalizer()
tokenizer = Tokenizer()
dictionary = Dictionary()

normalized_string = normalizer.normalize(string)
tokens = tokenizer.tokenize(normalized_string)

results = []
sentence_words = []
for token in tokens:
    result = dictionary.morphological_analysis(token)
    results.append(result)

print('-' * 50)
for i in range(len(results)):
    print(results[i])
    sentence_words.append(results[i]['Lemma'])


final_sentence = ' '.join(sentence_words)
print()
print('=' * 50)
print('=' * 50)
print('Sentence Words: ', final_sentence)