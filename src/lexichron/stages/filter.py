"""``lexichron filter``: the n-gram filter stage from plain settings."""
from __future__ import annotations

from typing import List, Optional, Tuple

from lexichron.resources import available_cpus

__all__ = ["filter_ngrams", "text_language_for", "combined_tokens_from_acquire", "source_is_untagged"]

# Languages in which a token's tag is used for its lemma (see
# ngramprep.ngram_filter.lemmatizer.SpacyLemmatizer.TAG_LANGUAGES).
TAG_LEMMA_LANGUAGES = ("en",)


def _check_vocabulary_lemmas(vocabulary: str, by_tag: bool, untagged_source: bool = False) -> None:
    """Refuse a vocabulary whose lemmas were made the other way.

    A vocabulary is a list of lemmas, and a token is kept only if its lemma
    is on it. Lemmas made with tags differ from the lookup's for some words
    ("building" the noun against "build"): filtering n-grams one way against
    a vocabulary built the other would silently drop every such token. The
    vocabulary's own record says how it was made; a file with no record
    beside it (one you supplied) is taken as it is."""
    from pathlib import Path
    from lexichron.provenance import read_provenance
    from ngramprep.ngram_filter.pipeline.orchestrator import LEMMAS_BY_TAG
    record = read_provenance(Path(vocabulary).parent)
    if not record or record.get("stage") != "filter":
        return
    if (record.get("settings") or {}).get("lemmatize", True) is False:
        return                                      # a vocabulary of surface forms: nothing to compare
    # The run records how it made its lemmas (its inputs' "lemmas"); a record
    # from before tags were used has no such entry, and used the lookup.
    made_by_tag = (record.get("inputs") or {}).get("lemmas") == LEMMAS_BY_TAG
    if made_by_tag != by_tag:
        how = {True: "with part-of-speech tags", False: "with the lookup table alone"}
        raise ValueError(
            f"The vocabulary {vocabulary} was built by lemmatizing {how[made_by_tag]}, and this run would "
            f"lemmatize {how[by_tag]}: tokens whose lemma differs between the two would not be found on it "
            f"and would be dropped. Run the unigram filter again so the vocabulary matches (it writes a new "
            f"whitelist), or "
            + ("use a vocabulary built from untagged unigrams: this run's source holds untagged n-grams, "
               "which have no tags to lemmatize by." if untagged_source and made_by_tag else
               f"set lemmatize_by_tag: {str(made_by_tag).lower()} for this run."))


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
    lemmatize_by_tag: bool = True,
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
    always_include_case: Optional[List[str]] = None,
    output_vocabulary: Optional[bool] = None,
    output_vocabulary_top_n: Optional[int] = 30_000,
    output_vocabulary_year_range: Optional[Tuple[int, int]] = None,
    output_vocabulary_spell_check: bool = False,
    workers: Optional[int] = None,
    mode: str = "resume",
    pivot: Optional[bool] = None,
    compact_after_ingest: bool = True,
    verify_counts: bool = True,
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
        lemmatize_by_tag: Use a token's part-of-speech tag when lemmatizing
            (English, tagged n-grams). The lookup table has one lemma per
            spelling, so the noun "building" and the verb both become
            "build", and "leaves" is "leave" whether on a tree or out of a
            room. With this on, a tagged noun, verb, adjective or adverb
            gets the lemma spaCy's tables give for that part of speech
            where they can vouch for it (an irregular form listed for it, or
            a known base form), and the lookup's lemma otherwise: building
            the noun stays "building", leaves the noun is "leaf", better the
            adjective is "good". Nothing is ever stripped from a word the
            tables do not know. A vocabulary must have been built the same
            way as the n-grams filtered against it, so turning this on or
            off means running the unigram filter again first; the stage
            refuses a vocabulary made the other way. False = the lookup
            alone, as before 0.12. A database of untagged n-grams
            (acquire's ngram_type: untagged) has no tags, so it gets the
            lookup alone whatever this says, and is recorded that way.
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
        always_include: Spellings kept whatever the other rules say, matched
            against the token as it appears in the data whatever its
            capitalization (he keeps He and HE; working-class keeps
            Working-Class). A kept token is still lowercased and lemmatized
            like any other; it only skips the vocabulary, alphabetic, length
            and stopword rules, so list every spelling you want kept (woman
            does not keep women). Tokens the acquire run combined
            (combined_multigrams) are added automatically from the raw
            database's provenance record, and those are not lemmatized
            either: they stay as the acquire run's entries made them
            A unigram run also writes these words to the vocabulary
            whatever their frequency, after the output_vocabulary_top_n
            most frequent, so listing a word there is enough: the longer
            n-grams filtered against that vocabulary keep it too
        always_include_case: Spellings kept only when their capitalization
            matches exactly (AI keeps AI and not ai)
        output_vocabulary: Write a vocabulary file from the result
            (<database>/whitelist.txt). Default: on for unigrams, off otherwise
        output_vocabulary_top_n: Keep the N most frequent tokens
        output_vocabulary_year_range: Only tokens present in every year of
            [start, end], e.g. [1968, 2019]
        output_vocabulary_spell_check: Only correctly spelled tokens (enchant)
        workers: Worker processes (default: CPUs - 1)
        mode: "resume" continues from the destination's markers; "restart"
            deletes the destination first
        pivot: Write <n>grams_pivoted.db (keyed year-first, one record per
            year and n-gram: what training reads) in one pass from the raw
            database, instead of <n>grams_processed.db (keyed by n-gram, a
            packed year list each: what the vocabulary is built from).
            Default: off for unigrams, on for longer n-grams
        compact_after_ingest: Ignored: a run that finished every work unit
            always compacts. Each unit arrives as its own SST file, and
            compaction is what folds them into one sorted database and sums
            the counts of records that arrived from more than one unit.
            Kept so that older settings files still load.
        verify_counts: Check the finished database against the run's own
            records (default True). Every work unit records the total
            occurrences and volume counts of what it read from the source
            and of what it emitted. After compaction the destination is
            read once, in parallel, and must hold exactly what the units
            emitted, with every value's years in order; and what the units
            read must add up to what the source database holds, where the
            acquisition that made it verified its own counts (a key range
            skipped or read twice by the partition shows there); and what
            the units read must equal what they kept plus what they
            dropped, each counted separately (a count lost in binning or in
            the pivot shows there). A difference fails the run. Turn it off only to save the pass on
            a database you will check another way.
        chunk_entries: Entries per chunk file (bounds each worker's memory)
        spool_dir: Where workers write chunk files (default: $TMPDIR)
        progress_path: Progress document for this run (set by the CLI)

    Returns:
        Path to the processed database directory
    """
    from ngramprep.ngram_filter.config import FilterConfig, PipelineConfig
    from lexichron.corpora import corpus_layout
    from ngramprep.ngram_filter.pipeline.orchestrator import build_processed_db

    lang = text_language or text_language_for(repo_corpus_id)
    if ascii_only is None:
        ascii_only = lang == "en"
    ngram_size = int(ngram_size)
    layout = corpus_layout(db_path_stub, ngram_size, repo_release_id, repo_corpus_id)
    if min_context_tokens is None:
        min_context_tokens = 1 if ngram_size == 1 else 2
    if pivot is None:
        pivot = ngram_size > 1
    output = "pivoted" if pivot else "processed"
    if output_vocabulary is None:
        output_vocabulary = not pivot and ngram_size == 1
    if output_vocabulary and pivot:
        raise ValueError("a vocabulary is built from an n-gram-keyed database; turn off pivot "
                         "or output_vocabulary")
    if no_vocabulary and vocabulary:
        raise ValueError("no_vocabulary is set but a vocabulary file is given as well; choose one")
    if vocabulary is None and ngram_size > 1 and not no_vocabulary:
        unigram = layout.shared_vocabulary
        if not unigram.exists():
            raise FileNotFoundError(
                f"{ngram_size}-grams are filtered against a vocabulary, and none was given or found at "
                f"{unigram}. Run the unigram filter for this corpus first, set vocabulary, "
                f"or set no_vocabulary to filter with the rules alone.")
        vocabulary = str(unigram)
        print(f"Vocabulary:   {vocabulary} (this corpus's unigram vocabulary)")
    by_tag = bool(lemmatize and lemmatize_by_tag and lang in TAG_LEMMA_LANGUAGES)
    # A source of untagged n-grams has no tags to use: every token gets the
    # lookup's lemma whatever the setting, and the run must say so, since a
    # vocabulary is matched to the n-grams by how their lemmas were made.
    untagged = by_tag and source_is_untagged(layout.raw_db)
    if untagged:
        by_tag = False
    if vocabulary and lemmatize:
        _check_vocabulary_lemmas(vocabulary, by_tag, untagged_source=untagged)

    stop_set = None
    if drop_stopwords and vocabulary is None:
        from ngramprep.ngram_filter.stopwords import load_stopwords
        stop_set, _ = load_stopwords(lang)
    lemma_gen = None
    if lemmatize:
        from ngramprep.ngram_filter.lemmatizer import CachedSpacyLemmatizer
        lemma_gen = CachedSpacyLemmatizer(language=lang, use_tags=by_tag)
        print(f"Lemmas:       {'by part-of-speech tag where the tables vouch for it, else the lookup table' if by_tag else 'lookup table'}"
              + (" (the source holds untagged n-grams, so there are no tags to use)" if untagged else ""))

    keep = [str(t) for t in (always_include or []) if str(t).strip()]
    handed = [t for t in combined_tokens_from_acquire(layout.raw_db) if t not in keep]
    if handed:
        print(f"Always keep:  {', '.join(handed)} (combined by the acquire run that made the raw database)")
        keep += handed
    keep_case = [str(t) for t in (always_include_case or []) if str(t).strip()]
    combined = combined_tokens_from_acquire(layout.raw_db)
    if combined and lemmatize:
        print(f"Not lemmatized: {', '.join(combined)} (tokens the acquire run combined are kept as they were made)")
    filter_config = FilterConfig(
        lowercase=lowercase, alpha_only=alpha_only, ascii_alpha_only=ascii_only,
        filter_short=min_len > 1, filter_stops=drop_stopwords and stop_set is not None,
        apply_lemmatization=lemmatize, min_len=int(min_len), min_context_tokens=int(min_context_tokens),
        stop_set=stop_set, stop_words_language=lang, lemma_gen=lemma_gen,
        whitelist_path=vocabulary, whitelist_top_n=vocabulary_top_n,
        always_include=set(keep) if keep else None,
        always_include_case=set(keep_case) if keep_case else None,
        lemma_exempt=set(combined) if combined else None,
        bin_size=int(bin_size),
    )
    pipeline_config = PipelineConfig(
        src_db=layout.raw_db, tmp_dir=layout.tmp_dir,
        num_workers=workers or max(1, available_cpus() - 1),
        num_initial_work_units=None, mode=mode, output=output,
        use_smart_partitioning=True,
        compact_after_ingest=compact_after_ingest,
        verify_counts=bool(verify_counts), source_totals=verified_source_totals(layout.raw_db),
        output_whitelist_path=layout.vocabulary if output_vocabulary else None,
        dst_db=layout.training_db if output == "pivoted" else layout.processed_db,
        output_whitelist_top_n=output_vocabulary_top_n,
        output_whitelist_year_range=tuple(output_vocabulary_year_range) if output_vocabulary_year_range else None,
        output_whitelist_spell_check=output_vocabulary_spell_check,
    )
    return build_processed_db(filter_config=filter_config, pipeline_config=pipeline_config,
                              progress_path=progress_path, spool_dir=spool_dir,
                              chunk_entries=int(chunk_entries))


def verified_source_totals(src_db) -> Optional[dict]:
    """``{"match_total", "volume_total"}`` of the source database, when the
    run that made it verified them (its provenance record's count check), or
    None. The filter's work units must read exactly this much."""
    from lexichron.provenance import read_provenance
    counts = ((read_provenance(src_db) or {}).get("figures") or {}).get("counts") or {}
    if counts.get("status") != "verified" or counts.get("occurrences") is None or counts.get("volume_counts") is None:
        return None
    return {"match_total": int(counts["occurrences"]), "volume_total": int(counts["volume_counts"])}


def source_is_untagged(src_db) -> bool:
    """Whether the acquire run that made ``src_db`` kept untagged n-grams
    only (its ``ngram_type``), from the database's provenance record. A
    database with no record, or one made with ``tagged`` or ``all``, may
    hold tags and is not called untagged."""
    from lexichron.provenance import read_provenance
    record = read_provenance(src_db) or {}
    if record.get("stage") not in (None, "acquire"):
        return False
    return (record.get("settings") or {}).get("ngram_type") == "untagged"


def combined_tokens_from_acquire(src_db) -> List[str]:
    """The tokens the acquire run that made ``src_db`` built from word
    sequences (its combined_multigrams), read from the database's provenance
    record: each entry's target, or its words joined by hyphens. Empty when
    the database has no record or the run combined nothing. The filter
    stage keeps these automatically, since nothing else could: a unigram
    vocabulary cannot contain them and the alphabetic rule would drop them.
    It does not lemmatize them either: each was made to the user's
    specification, and stays as made (lowercased with everything else)."""
    from lexichron.provenance import read_provenance
    from ngramprep.ngram_acquire.io.parse import parse_multigram_entry
    record = read_provenance(src_db) or {}
    out: List[str] = []
    settings = record.get("settings") or {}
    # both lists: the sequences matched whatever their case, and those matched as written
    for entry in list(settings.get("combined_multigrams") or []) + list(settings.get("combined_multigrams_case") or []):
        try:
            words, target = parse_multigram_entry(entry)
        except Exception:
            continue
        token = target or "-".join(words)
        if token and token not in out:
            out.append(token)
    return out
