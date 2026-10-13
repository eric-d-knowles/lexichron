"""``lexichron filter``: the n-gram filter stage from plain settings."""
from __future__ import annotations

from typing import List, Optional, Tuple

__all__ = ["filter_ngrams", "text_language_for"]

# Google Books corpus id -> spaCy language code (stopwords and lemmatizer).
_CORPUS_LANGUAGE = {
    "eng": "en", "eng-us": "en", "eng-gb": "en", "eng-fiction": "en",
    "fre": "fr", "ger": "de", "ita": "it", "spa": "es", "rus": "ru", "heb": "he", "chi-sim": "zh",
}


def text_language_for(repo_corpus_id: str) -> str:
    try:
        return _CORPUS_LANGUAGE[repo_corpus_id]
    except KeyError:
        raise ValueError(f"no text language known for corpus {repo_corpus_id!r}; set text_language")


def filter_ngrams(
    ngram_size: int,
    repo_release_id: str,
    repo_corpus_id: str,
    db_path_stub: str,
    text_language: Optional[str] = None,
    lowercase: bool = True,
    lemmatize: bool = True,
    drop_stopwords: bool = True,
    alpha_only: bool = True,
    ascii_only: Optional[bool] = None,
    min_len: int = 3,
    min_context_tokens: Optional[int] = None,
    bin_size: int = 1,
    vocabulary: Optional[str] = None,
    no_vocabulary: bool = False,
    vocabulary_top_n: Optional[int] = None,
    always_include: Optional[List[str]] = None,
    output_vocabulary: Optional[bool] = None,
    output_vocabulary_top_n: Optional[int] = 30_000,
    output_vocabulary_year_range: Optional[Tuple[int, int]] = None,
    output_vocabulary_spell_check: bool = False,
    workers: Optional[int] = None,
    mode: str = "resume",
    output: Optional[str] = None,
    compact_after_ingest: bool = True,
    chunk_entries: int = 200_000,
    spool_dir: Optional[str] = None,
    progress_path: Optional[str] = None,
) -> str:
    """
    Filter a raw n-gram database. Unigrams become a processed database (and
    the vocabulary the later stages use); longer n-grams become, in the same
    pass, the pivoted (year-keyed) database that training reads.

    Args:
        ngram_size: N-gram size (1-5)
        repo_release_id: Release date in YYYYMMDD format (e.g., "20200217")
        repo_corpus_id: Corpus identifier (e.g., "eng", "eng-us")
        db_path_stub: Base directory for databases
        text_language: Language of the text for stopwords and lemmatization
            (spaCy code: en, fr, de, it, es, ru, he, zh). Default: derived
            from the corpus (eng-us -> en).
        lowercase: Lowercase every token
        lemmatize: Reduce tokens to their lemma (women -> woman) using spaCy's
            lookup tables for text_language. Also applied before a vocabulary
            lookup, since vocabularies are lists of lemmas.
        drop_stopwords: Replace stopwords of text_language with <UNK>
        alpha_only: Replace tokens that are not purely alphabetic with <UNK>
        ascii_only: With alpha_only, also reject letters outside A-Z. Default:
            on for English, off for other languages
        min_len: Replace tokens shorter than this with <UNK>
        min_context_tokens: Drop n-grams with fewer than this many non-<UNK>
            tokens. Default: 1 for unigrams, 2 for longer n-grams
        bin_size: Aggregate years into bins (1 = annual, 5 = five-year bins)
        vocabulary: Vocabulary file (one token per line, counts optional):
            tokens not in it become <UNK> and the other filters are skipped.
            Unigrams use none. Longer n-grams default to this corpus's own
            unigram vocabulary (<...>/1gram_files/1grams_processed.db/whitelist.txt),
            which the unigram filter writes; the stage refuses to run without one
            unless no_vocabulary is set.
        no_vocabulary: Filter longer n-grams with the rule filters alone, with
            no vocabulary. Rarely wanted: the result is far larger.
        vocabulary_top_n: Use only the first N tokens of the vocabulary file
        always_include: Tokens kept even if not in the vocabulary (e.g. focal
            terms the unigram stage dropped as stopwords: he, she, ...)
        output_vocabulary: Write a vocabulary file from the result
            (<database>/whitelist.txt). Default: on for unigrams, off otherwise
        output_vocabulary_top_n: Keep the N most frequent tokens
        output_vocabulary_year_range: Only tokens present in every year of
            [start, end], e.g. [1968, 2019]
        output_vocabulary_spell_check: Only correctly spelled tokens (enchant)
        workers: Worker processes (default: CPUs - 1)
        mode: "resume" continues from the destination's markers; "restart"
            deletes the destination first
        output: "processed" writes <n>grams_processed.db (keyed by n-gram, a
            packed year list each; the vocabulary is built from it) or
            "pivoted" writes <n>grams_pivoted.db (keyed year-first, one
            record per year and n-gram, what training reads) in one pass
            from the raw database. Default: processed for unigrams, pivoted
            for longer n-grams
        compact_after_ingest: Compact the destination at the end
        chunk_entries: Entries per chunk file (bounds each worker's memory)
        spool_dir: Where workers write chunk files (default: $TMPDIR)
        progress_path: Progress document for this run (set by the CLI)

    Returns:
        Path to the processed database directory
    """
    from ngramprep.ngram_filter.config import FilterConfig, PipelineConfig
    from ngramprep.ngram_filter.pipeline.orchestrator import build_processed_db, derive_paths

    lang = text_language or text_language_for(repo_corpus_id)
    if ascii_only is None:
        ascii_only = lang == "en"
    ngram_size = int(ngram_size)
    if min_context_tokens is None:
        min_context_tokens = 1 if ngram_size == 1 else 2
    if output is None:
        output = "processed" if ngram_size == 1 else "pivoted"
    if output not in ("processed", "pivoted"):
        raise ValueError(f"output must be 'processed' or 'pivoted', not {output!r}")
    if output_vocabulary is None:
        output_vocabulary = output == "processed" and ngram_size == 1
    if output_vocabulary and output == "pivoted":
        raise ValueError("a vocabulary is built from a processed database; set output to processed "
                         "or turn off output_vocabulary")
    if no_vocabulary and vocabulary:
        raise ValueError("no_vocabulary is set but a vocabulary file is given as well; choose one")
    if vocabulary is None and ngram_size > 1 and not no_vocabulary:
        unigram = derive_paths(db_path_stub, 1, str(repo_release_id), str(repo_corpus_id))["whitelist"]
        if not unigram.exists():
            raise FileNotFoundError(
                f"{ngram_size}-grams are filtered against a vocabulary, and none was given or found at "
                f"{unigram}. Run the unigram filter for this corpus first, set vocabulary, "
                f"or set no_vocabulary to filter with the rules alone.")
        vocabulary = str(unigram)
        print(f"Vocabulary:   {vocabulary} (this corpus's unigram vocabulary)")

    stop_set = None
    if drop_stopwords and vocabulary is None:
        from ngramprep.ngram_filter.stopwords import load_stopwords
        stop_set, _ = load_stopwords(lang)
    lemma_gen = None
    if lemmatize:
        from ngramprep.ngram_filter.lemmatizer import CachedSpacyLemmatizer
        lemma_gen = CachedSpacyLemmatizer(language=lang)

    filter_config = FilterConfig(
        lowercase=lowercase, alpha_only=alpha_only, ascii_alpha_only=ascii_only,
        filter_short=min_len > 1, filter_stops=drop_stopwords and stop_set is not None,
        apply_lemmatization=lemmatize, min_len=int(min_len), min_context_tokens=int(min_context_tokens),
        stop_set=stop_set, stop_words_language=lang, lemma_gen=lemma_gen,
        whitelist_path=vocabulary, whitelist_top_n=vocabulary_top_n,
        always_include=set(always_include) if always_include else None,
        bin_size=int(bin_size),
    )
    paths = derive_paths(db_path_stub, ngram_size, str(repo_release_id), str(repo_corpus_id))
    pipeline_config = PipelineConfig(
        src_db=paths["src_db"], tmp_dir=paths["tmp_dir"],
        num_workers=workers or max(1, (__import__("os").cpu_count() or 2) - 1),
        num_initial_work_units=None, mode=mode, output=output,
        use_smart_partitioning=True,
        compact_after_ingest=compact_after_ingest,
        output_whitelist_path=paths["whitelist"] if output_vocabulary else None,
        dst_db=paths["pivoted_db"] if output == "pivoted" else paths["dst_db"],
        output_whitelist_top_n=output_vocabulary_top_n,
        output_whitelist_year_range=tuple(output_vocabulary_year_range) if output_vocabulary_year_range else None,
        output_whitelist_spell_check=output_vocabulary_spell_check,
    )
    return build_processed_db(filter_config=filter_config, pipeline_config=pipeline_config,
                              progress_path=progress_path, spool_dir=spool_dir,
                              chunk_entries=int(chunk_entries))
