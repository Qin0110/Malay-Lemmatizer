import re

import enchant


class LanguageIdentifier:
    """Decide, per token, whether to lemmatize as English or as Malay.

    Enchant alone cannot make this call: "mana", "pun", "lain", "main",
    "kami" and "orang" are accepted by the English dictionary *and* exist in
    MALINDO. Routing purely on enchant sends those Malay words to WordNet, so
    a token only counts as English when enchant accepts it and the Malay
    lexicon does not claim it.
    """

    # Elongation ("gilaa", "sedapp") and the "budak2" reduplication shorthand
    # never survive a dictionary lookup, so strip them before deciding.
    ELONGATION_RE = re.compile(r"(.)\1{2,}")
    REDUP_RE = re.compile(r"^(.+?)2$")

    def __init__(self, malay_lexicon=None, lang="en_US"):
        self.dictionary = enchant.Dict(lang)
        self.malay_lexicon = malay_lexicon or set()

    def _variants(self, token):
        """Surface forms worth testing, cheapest first."""
        token = token.lower()
        yield token

        redup = self.REDUP_RE.match(token)
        if redup:
            yield redup.group(1)

        # "gilaaa" -> "gilaa" -> "gila": collapse runs to two, then to one.
        if self.ELONGATION_RE.search(token):
            yield self.ELONGATION_RE.sub(r"\1\1", token)
            yield self.ELONGATION_RE.sub(r"\1", token)

    def is_malay(self, token):
        return any(v in self.malay_lexicon for v in self._variants(token))

    def is_english(self, token):
        # Enchant accepts single letters and bare digits; those are noise here.
        if len(token) < 2 or not token.isalpha():
            return False
        return any(self.dictionary.check(v) for v in self._variants(token))

    def identify(self, token):
        """Return "en", "ms" or "unknown" for a single token."""
        if not token or not token.strip():
            return "unknown"

        malay = self.is_malay(token)
        english = self.is_english(token)

        # Malay wins every overlap: this is a Malay lemmatizer, and the rojak
        # input is Malay-matrix text with English borrowings, not the reverse.
        if malay:
            return "ms"
        if english:
            return "en"
        return "unknown"

    def identify_all(self, tokens):
        return [(token, self.identify(token)) for token in tokens]
