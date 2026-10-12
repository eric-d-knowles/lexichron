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
    min_context_tokens: int = 1,
    bin_size: int = 1,
    vocabulary: Optional[str] = None,
    vocabulary_top_n: Optional[int] = None,
    always_include: Optional[List[str]] = None,
    output_vocabulary: bool = True,
    output_vocabulary_top_n: Optional[int] = 30_000,
    output_vocabulary_year_range: Optional[Tuple[int, int]] = None,
    output_vocabulary_spell_check: bool = False,
    workers: Optional[int] = None,
    mode: str = "resume",
    work_units: Optional[int] = None,
    compact_after_ingest: bool = True,
    smart_partitioning: bool = True,
    samples_per_worker: int = 50_000,
    chunk_entries: int = 200_000,
    spool_dir: Optional[str] = None,
    progress_path: Optional[str] = None,
) -> str:
    """
    Filter a raw n-gram database into a processed one, and (for unigrams)
    write the vocabulary the later stages use.

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
        min_context_tokens: Drop n-grams with fewer than this many non-<UNK> tokens
        bin_size: Aggregate years into bins (1 = annual, 5 = five-year bins)
        vocabulary: Vocabulary file (one token per line, counts optional):
            tokens not in it become <UNK> and the other filters are skipped.
            Typically the unigram stage's output for the same corpus.
        vocabulary_top_n: Use only the first N tokens of the vocabulary file
        always_include: Tokens kept even if not in the vocabulary (e.g. focal
            terms the unigram stage dropped as stopwords: he, she, ...)
        output_vocabulary: Write a vocabulary file from the result
            (<database>/whitelist.txt). Meant for unigrams.
        output_vocabulary_top_n: Keep the N most frequent tokens
        output_vocabulary_year_range: Only tokens present in every year of
            [start, end], e.g. [1968, 2019]
        output_vocabulary_spell_check: Only correctly spelled tokens (enchant)
        workers: Worker processes (default: CPUs - 1)
        mode: "resume" continues from the destination's markers; "restart"
            deletes the destination first
        work_units: Number of key ranges to split the work into (default:
            4 x workers, at least 64; more = better balance)
        compact_after_ingest: Compact the destination at the end
        smart_partitioning: Balance the key ranges by sampling the source
        samples_per_worker: Reservoir size per sampling worker
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
    paths = derive_paths(db_path_stub, int(ngram_size), str(repo_release_id), str(repo_corpus_id))
    pipeline_config = PipelineConfig(
        src_db=paths["src_db"], dst_db=paths["dst_db"], tmp_dir=paths["tmp_dir"],
        num_workers=workers or max(1, (__import__("os").cpu_count() or 2) - 1),
        num_initial_work_units=work_units, mode=mode,
        use_smart_partitioning=smart_partitioning, samples_per_worker=int(samples_per_worker),
        compact_after_ingest=compact_after_ingest,
        output_whitelist_path=paths["whitelist"] if output_vocabulary else None,
        output_whitelist_top_n=output_vocabulary_top_n,
        output_whitelist_year_range=tuple(output_vocabulary_year_range) if output_vocabulary_year_range else None,
        output_whitelist_spell_check=output_vocabulary_spell_check,
    )
    return build_processed_db(filter_config=filter_config, pipeline_config=pipeline_config,
                              progress_path=progress_path, spool_dir=spool_dir,
                              chunk_entries=int(chunk_entries))
