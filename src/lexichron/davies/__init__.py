"""Reading Mark Davies' corpora (COHA first) for the acquire stage.

The corpora are licensed and held by the user; nothing here downloads
anything. :mod:`formats` says how each corpus's files are laid out,
:mod:`reader` turns a file into documents and documents into units of
running text, and :mod:`worker` is what one acquire worker does with one
archive. Everything after the worker (ingest with the marker, compaction,
the count check) is the n-gram acquisition's own
(:func:`ngramprep.ngram_acquire.core.run_acquisition`).
"""
