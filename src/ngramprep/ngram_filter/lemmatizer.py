"""
Lemmatizer wrapper for spaCy.

Provides a simple .lemmatize(word, pos) interface compatible with the Cython filter pipeline.
"""
import logging
import spacy
from functools import lru_cache


# Default spaCy models by language
DEFAULT_MODELS = {
    "en": "en_core_web_sm",      # English
    "de": "de_core_news_sm",     # German (ger)
    "es": "es_core_news_sm",     # Spanish (spa)
    "fr": "fr_core_news_sm",     # French (fre)
    "it": "it_core_news_sm",     # Italian (ita)
    "ru": "ru_core_news_sm",     # Russian (rus)
    "zh": "zh_core_web_sm",      # Chinese (chi-sim)
    "he": "he_core_news_sm",     # Hebrew (heb)
}


class SpacyLemmatizer:
    """
    Lemmatizes single tokens with spaCy's tables, using a token's part of
    speech where there is one and the tables can vouch for the answer.

    Two candidates are weighed for a tagged token:

    * the *lookup* lemma: spaCy's lookup table gives one lemma per spelling,
      whatever its part of speech. It never invents a form, but it cannot
      tell the noun *building* from the verb (both become ``build``), nor
      *leaves* of a tree from *leaves* the room.
    * the *tag-based* lemmas: what spaCy's rule tables allow for this
      spelling **as this part of speech**: an irregular form listed for it
      (``leaves`` as a noun is ``leaf``; ``better`` as an adjective is
      ``good``), the word itself when it is a known base form (``building``
      is a noun in its own right), and the results of the suffix rules that
      are themselves known base forms (``buildings`` -> ``building``).

    spaCy's own rule lemmatizer is not used as it stands: given only a coarse
    tag it strips suffixes from words it does not recognize, and never asks
    whether the word is already a base form (``business`` -> ``busines``).
    Here a tag-based lemma is taken only when the tables vouch for it:

    1. an irregular form listed for the part of speech wins;
    2. otherwise, if the lookup lemma is among the tag-based ones, it stands
       (the two agree: ``years`` -> ``year``, ``glasses`` -> ``glass``);
    3. otherwise the lookup lemma is not a lemma this part of speech can
       have, and the tag-based one is taken: the word itself if it is a base
       form (``building`` the noun stays ``building``), else the first rule
       result;
    4. with no tag-based lemma at all (an unknown word, a part of speech
       without tables, no tag) the lookup decides, as it always did.

    So the tag changes an answer only when the lookup's answer is not a
    legitimate lemma for the token's part of speech, or an irregular form
    says otherwise. Tags are used for nouns, verbs, adjectives and adverbs,
    in English (``TAG_LANGUAGES``); for other languages, and with
    ``use_tags=False``, this is the lookup alone.

    Usage:
        lemmatizer = SpacyLemmatizer()
        lemmatizer.lemmatize("building", pos="NOUN")   # "building"
        lemmatizer.lemmatize("building", pos="VERB")   # "build"
        lemmatizer.lemmatize("building")               # "build" (the lookup)
    """

    TAGGED = ("noun", "verb", "adj", "adv")
    # Languages whose rule tables have been checked against real data for
    # this use. Other models may carry tables of the same names with other
    # conventions; until they are checked, their tokens get the lookup alone.
    TAG_LANGUAGES = ("en",)

    def __init__(self, language=None, model=None, use_tags=True):
        """
        Initialize the spaCy lemmatizer.

        Args:
            language: ISO 639-1 language code (e.g., "en", "de", "fr"). Defaults to "en".
            model: Explicit spaCy model name. Overrides language parameter if provided.
            use_tags: Use a token's part of speech where the tables allow
                (see the class docstring). False = the lookup table alone.
        """
        if model is None:
            lang = language or "en"
            if lang not in DEFAULT_MODELS:
                raise ValueError(
                    f"Language '{lang}' not supported. "
                    f"Supported languages: {', '.join(DEFAULT_MODELS.keys())}. "
                    f"Or specify a custom model with the 'model' parameter."
                )
            model = DEFAULT_MODELS[lang]

        self.nlp = spacy.load(model, disable=["parser", "ner", "textcat"])

        # The model's own lemmatizer is kept for its rule tables (the index of
        # base forms, the irregular forms, the suffix rules); it is never run.
        self.rule_lemmatizer = self.nlp.get_pipe("lemmatizer")

        # Remove it and add lookup-based lemmatizer as primary
        self.nlp.remove_pipe("lemmatizer")
        self.nlp.add_pipe("lemmatizer", name="lemmatizer_lookup", config={"mode": "lookup"})

        # Suppress spaCy's INFO messages during initialization
        spacy_logger = logging.getLogger("spacy")
        original_level = spacy_logger.level
        spacy_logger.setLevel(logging.WARNING)

        self.nlp.initialize()

        # Restore original logging level
        spacy_logger.setLevel(original_level)

        self._lookup_table = self.nlp.get_pipe("lemmatizer_lookup").lookups.get_table("lemma_lookup")
        self.use_tags = bool(use_tags)
        self._tag_tables = self._load_tag_tables() if self.use_tags else {}

    def _load_tag_tables(self) -> dict:
        """``{pos: (irregular forms, base forms, suffix rules)}`` as plain
        dicts and sets, for the parts of speech the model has all three
        tables for. Empty when the model's lemmatizer is not rule-based, or
        its language is not one of ``TAG_LANGUAGES``."""
        if self.nlp.lang not in self.TAG_LANGUAGES:
            return {}
        try:
            lookups = self.rule_lemmatizer.lookups
            index, exc, rules = (lookups.get_table(name) for name in ("lemma_index", "lemma_exc", "lemma_rules"))
        except Exception:
            return {}
        tables = {}
        for pos in self.TAGGED:
            base_forms = index.get(pos)
            if not base_forms:
                continue
            irregular = {str(word): str(forms[0]) for word, forms in (exc.get(pos) or {}).items() if forms}
            tables[pos] = (irregular, set(map(str, base_forms)),
                           [(str(old), str(new)) for old, new in (rules.get(pos) or [])])
        return tables

    def lookup(self, word: str) -> str:
        """The lookup table's lemma for this spelling: one answer whatever the
        part of speech, and the word itself when the table does not list it
        (no suffix is ever stripped from an unknown word)."""
        return str(self._lookup_table.get(word, word))

    def tag_lemmas(self, word: str, pos) -> tuple:
        """``(irregular, others)`` for the word as this part of speech: the
        irregular form listed for it (or None), and the other lemmas the
        tables vouch for, the word itself first if it is a base form, then
        the suffix-rule results that are base forms. ``(None, [])`` when the
        tables have nothing to say."""
        tables = self._tag_tables.get(str(pos).lower()) if pos else None
        if tables is None or word != word.lower():       # the tables are lowercase; a cased token is left to the lookup
            return None, []
        irregular, base_forms, rules = tables
        others = [word] if word in base_forms else []
        for old, new in rules:
            if old and word.endswith(old):
                form = word[: len(word) - len(old)] + new
                if form and form in base_forms and form not in others:
                    others.append(form)
        return irregular.get(word), others

    def _lemmatize(self, word: str, pos) -> str:
        looked_up = self.lookup(word)
        if not pos or not self._tag_tables:
            return looked_up
        irregular, others = self.tag_lemmas(word, pos)
        if irregular is not None:
            return irregular
        if not others or looked_up in others:
            return looked_up
        return others[0]

    def lemmatize(self, word: str, pos=None) -> str:
        """
        Lemmatize a single word, using its part of speech if one is given.

        Args:
            word: The word to lemmatize
            pos: Its part of speech (NOUN, VERB, ADJ, ADV; anything else, or
                None, means "no usable tag": the lookup decides)

        Returns:
            The lemma, or the word itself where the tables do not know it
        """
        return self._lemmatize(word, pos)


class CachedSpacyLemmatizer(SpacyLemmatizer):
    """
    Cached version of SpacyLemmatizer for better performance.

    Uses LRU cache to avoid re-lemmatizing the same (word, pos) pairs.
    Recommended for processing large corpora with repeated tokens.

    Usage:
        lemmatizer = CachedSpacyLemmatizer(cache_size=100000)
        result = lemmatizer.lemmatize("running", pos="VERB")  # cached
    """

    def __init__(self, language=None, model=None, cache_size=100000, use_tags=True):
        """
        Initialize cached lemmatizer.

        Args:
            language: ISO 639-1 language code
            model: Explicit spaCy model name
            cache_size: Maximum number of (word, pos) pairs to cache
            use_tags: Use a token's part of speech where the tables allow
        """
        # Store config for lazy loading (avoids pickling the model)
        self._language = language
        self._model = model
        self._cache_size = cache_size
        self._use_tags = bool(use_tags)
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy-initialize the spaCy model on first use in each process."""
        if not self._initialized:
            super().__init__(language=self._language, model=self._model, use_tags=self._use_tags)
            self._cached_lemmatize = lru_cache(maxsize=self._cache_size)(self._lemmatize)
            self._initialized = True

    def lemmatize(self, word: str, pos=None) -> str:
        """Lemmatize with caching."""
        self._ensure_initialized()
        return self._cached_lemmatize(word, pos)

    def lookup(self, word: str) -> str:
        self._ensure_initialized()
        return super().lookup(word)

    def tag_lemmas(self, word: str, pos) -> tuple:
        self._ensure_initialized()
        return super().tag_lemmas(word, pos)

    def cache_info(self):
        """Return cache statistics."""
        self._ensure_initialized()
        return self._cached_lemmatize.cache_info()

    def clear_cache(self):
        """Clear the lemmatization cache."""
        if self._initialized:
            self._cached_lemmatize.cache_clear()

    def __getstate__(self):
        """Custom pickling - only pickle config, not the loaded model."""
        return {
            '_language': self._language,
            '_model': self._model,
            '_cache_size': self._cache_size,
            '_use_tags': getattr(self, '_use_tags', True),
            '_initialized': False  # Force reload in each process
        }

    def __setstate__(self, state):
        """Custom unpickling - just restore config, model loads on first use."""
        self.__dict__.update(state)
