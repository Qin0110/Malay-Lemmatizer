from malindo_morph.dictionary import Dictionary
from Preprocess.english_lemmatizer import EnglishLemmatizer
from Preprocess.language_identifier import LanguageIdentifier
from Preprocess.tokenizer import Tokenizer


class HybridLemmatizer:
    """Route each token to the lemmatizer that matches its language.

    Malay tokens keep the existing MALINDO morphological analysis; English
    tokens go to WordNet instead. Unknown tokens fall back to the Malay path,
    which already returns the token unchanged when it cannot resolve one.
    """

    # Tokens neither lemmatizer should touch. WordNet reads the pronoun "us" as
    # a plural noun and returns "u"; MALINDO reads the ethnonym "melayu" as
    # me-+layu and returns "layu" ("to wilt"). Both are lookup artefacts, not
    # morphology, and both appear in ordinary rojak text.
    KEEP_AS_IS = {
        "us", "as", "its", "his", "hers", "ours", "yours", "theirs",
        "melayu", "malaysia", "bahasa",
    }

    def __init__(self, normalizer=None, dictionary=None):
        # The Malay normalizer loads a KenLM model and two HuggingFace models,
        # so it stays optional: callers that only need routing skip that cost.
        self.normalizer = normalizer
        self.tokenizer = Tokenizer()
        self.dictionary = dictionary or Dictionary()
        self.english = EnglishLemmatizer()
        self.identifier = LanguageIdentifier(
            malay_lexicon=self.dictionary.surface_lexicon()
        )

    def analyze(self, sentence):
        """Return one record per token: language, lemma and Malay morphology."""
        if self.normalizer is not None:
            sentence = self.normalizer.normalize(sentence)
        tokens = self.tokenizer.tokenize(sentence)
        tokens = [t for t in tokens if t]
        if not tokens:
            return []

        languages = [self.identifier.identify(t) for t in tokens]
        for i, token in enumerate(tokens):
            if token.lower() in self.KEEP_AS_IS:
                languages[i] = "keep"

        # MALINDO also lists English borrowings ("the", "real", "margin"), so a
        # token can be Malay-by-lexicon yet English in context. That is usually
        # harmless -- the Malay path returns those unchanged -- but "at" is a
        # MALINDO entry whose root is "atas", which would rewrite an English
        # word into a Malay one. So for tokens the English dictionary also
        # accepts, check what the Malay path would do first, and hand the token
        # to English only when Malay would actually rewrite it.
        for i, (token, language) in enumerate(zip(tokens, languages)):
            if language != "ms" or not self.identifier.is_english(token):
                continue
            malay_lemma = self.dictionary.morphological_analysis(token)["Lemma"]
            if malay_lemma.lower() != token.lower():
                languages[i] = "en"

        # Tag the English tokens as one sequence so the POS tagger still sees
        # the words around each one, rather than tagging them in isolation.
        english_index = [i for i, lang in enumerate(languages) if lang == "en"]
        english_lemmas = dict(
            zip(
                english_index,
                self.english.lemmatize_tokens([tokens[i] for i in english_index]),
            )
        )

        records = []
        for i, (token, language) in enumerate(zip(tokens, languages)):
            if language == "keep":
                records.append(
                    {
                        "Word": token,
                        "Language": "keep",
                        "Lemma": token,
                        "Candidates": [],
                    }
                )
            elif language == "en":
                records.append(
                    {
                        "Word": token,
                        "Language": "en",
                        "Lemma": english_lemmas[i],
                        "Candidates": [],
                    }
                )
            else:
                record = self.dictionary.morphological_analysis(token)
                record["Language"] = language
                records.append(record)
        return records

    def lemmatize(self, sentence):
        """Lemmatize a sentence and return it as a string."""
        return " ".join(r["Lemma"].lower() for r in self.analyze(sentence))
