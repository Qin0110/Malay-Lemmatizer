import re

class Tokenizer:
    def __init__(self):
        pass
    
    def tokenize(self, sentences):
        result = re.sub(r'[^\w\s]', '', sentences)
        result = re.split(r'\s+', result)
        return result                       # return array