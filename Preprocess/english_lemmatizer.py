import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


class EnglishLemmatizer:
    """WordNet lemmatizer with POS tagging.

    WordNetLemmatizer defaults to the noun tag, which leaves verbs and
    adjectives untouched ("running" stays "running"). Tagging the sentence
    first is what makes "running" -> "run" and "better" -> "good" work.
    """

    # Penn Treebank tag prefix -> WordNet POS constant.
    POS_MAP = {
        "J": wordnet.ADJ,
        "V": wordnet.VERB,
        "N": wordnet.NOUN,
        "R": wordnet.ADV,
    }

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self._ensure_data()

    @staticmethod
    def _ensure_data():
        required = [
            ("corpora/wordnet", "wordnet"),
            ("corpora/omw-1.4", "omw-1.4"),
            ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
        ]
        for path, package in required:
            try:
                nltk.data.find(path)
            except LookupError:
                nltk.download(package, quiet=True)

    def _wordnet_pos(self, treebank_tag):
        return self.POS_MAP.get(treebank_tag[:1].upper(), wordnet.NOUN)

    def _lemmatize_tagged(self, token, pos):
        """Lemmatize with the tagged POS, falling back to noun then verb.

        Rojak text hands the tagger Malay-English fragments, so tags are
        unreliable ("laptops" tags as JJ in isolation, NNS in a sentence). An
        adjective lookup leaves a plural noun untouched, so when the tagged
        POS is a no-op, retry as a noun.

        Only the noun retry is safe. Retrying as a verb would also turn the
        borrowed noun "meeting" into "meet", and rojak borrows English nouns
        far more often than it borrows inflected verbs.
        """
        token = token.lower()
        lemma = self.lemmatizer.lemmatize(token, pos)
        if lemma != token:
            return lemma
        if pos != wordnet.NOUN:
            return self.lemmatizer.lemmatize(token, wordnet.NOUN)
        return token

    def lemmatize(self, token, pos=None):
        """Lemmatize one token; POS is guessed from the token alone if absent."""
        if pos is not None:
            return self.lemmatizer.lemmatize(token.lower(), pos)
        tag = self._wordnet_pos(nltk.pos_tag([token])[0][1])
        return self._lemmatize_tagged(token, tag)

    def lemmatize_tokens(self, tokens):
        """Lemmatize a token list, tagging them together to keep context.

        Tagging the whole list beats tagging word-by-word, because the tagger
        uses surrounding words to tell a noun from a verb.
        """
        if not tokens:
            return []
        tagged = nltk.pos_tag(tokens)
        return [
            self._lemmatize_tagged(token, self._wordnet_pos(tag))
            for token, tag in tagged
        ]
