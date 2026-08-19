import re
from cleantext import clean

class Tokenizer:
    def __init__(self):
        pass
    
    def tokenize(self, sentences):
        result = re.sub(r'[^\w\s]', '', sentences)
        result = clean(result, no_emoji=True)
        result = re.sub(r"#[A-Za-z0-9_]+|@[A-Za-z0-9_]+|\n|http\S+|\"", "", result)
        result = re.split(r'\s+', result.strip())
        return result                       # return array