import re
from cleantext import clean

# Malay writes every kata ganda with a hyphen -- "rama-rama", "kanak-kanak",
# "berjalan-jalan" -- and MALINDO stores those surface forms with the hyphen in
# place. A blanket [^\w\s] strip turns "rama-rama" into "ramarama", which
# matches no entry, so the token falls through as OOV and the reduplication is
# lost. Keep the hyphen that joins two word characters and drop the rest:
# branch 1 captures the hyphen to protect, branch 2 matches punctuation to
# delete, and group(1) tells them apart.
PUNCTUATION_RE = re.compile(r'((?<=\w)-(?=\w))|[^\w\s]')


class Tokenizer:
    def __init__(self):
        pass

    def tokenize(self, sentences):
        result = PUNCTUATION_RE.sub(lambda m: m.group(1) or '', sentences)
        result = clean(result, no_emoji=True)
        result = re.sub(r"#[A-Za-z0-9_]+|@[A-Za-z0-9_]+|\n|http\S+|\"", "", result)
        result = re.split(r'\s+', result.strip())
        return result                       # return array
