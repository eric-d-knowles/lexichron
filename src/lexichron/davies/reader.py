"""From an archive to documents, and from a document to units of text.

A *document* is one text of the corpus with its year and genre. A *unit*
is a stretch of it whose words really were next to each other: a sentence,
or less where the corpus interrupts one. Units end at sentence-final
punctuation (``.``, ``?``, ``!`` as tokens of their own), at paragraph and
headline marks, and wherever the corpus removed words for copyright; none
of those tokens is kept. The rule looks only at the words, never at the
tags, so the tagged and the untagged edition are cut the same way.
"""
from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

from lexichron.davies.formats import EDITIONS, CorpusFormat, google_tag

__all__ = ["Document", "archives", "documents", "units", "SENTENCE_END", "ReadError"]

SENTENCE_END = frozenset({".", "?", "!"})


class ReadError(ValueError):
    """A file that is not laid out as its corpus's format says. The archive
    it is in fails as a whole: nothing of a file that cannot be read in the
    expected way is taken on trust."""


class Document:
    """One text: where and when it is from, and its tokens as written.
    ``tokens`` are the words; ``tags`` the coarse tag of each (None where
    the row had none), or None for the untagged edition. ``unreadable``
    counts rows of the tagged edition that were not word, lemma, tag."""
    __slots__ = ("name", "year", "genre", "tokens", "tags", "unreadable", "bad_rows", "nbytes")

    def __init__(self, name: str, year: int, genre: str, tokens: List[str], tags: Optional[List[Optional[str]]],
                 unreadable: int = 0, bad_rows: Sequence[str] = (), nbytes: int = 0) -> None:
        self.name, self.year, self.genre = name, year, genre
        self.tokens, self.tags = tokens, tags
        self.unreadable, self.bad_rows, self.nbytes = unreadable, list(bad_rows), nbytes


def archives(source_dir: str | Path, fmt: CorpusFormat, ngram_type: str) -> List[Path]:
    """The archives of the edition ``ngram_type`` reads, in name order.
    ``source_dir`` is the corpus's own directory, the one holding ``text/``
    and ``wlp/``."""
    if ngram_type not in EDITIONS:
        raise ValueError(f"ngram_type must be one of {tuple(EDITIONS)}, got {ngram_type!r}")
    edition = EDITIONS[ngram_type]
    root = Path(source_dir) / edition
    if not root.is_dir():
        other = [e for e in EDITIONS.values() if (Path(source_dir) / e).is_dir()]
        hint = (f" It has the {'untagged' if other[0] == 'text' else 'tagged'} edition ({other[0]}/): "
                f"set ngram_type accordingly to read that.") if other else ""
        raise FileNotFoundError(f"{fmt.label}'s {ngram_type} edition was not found: {root} is not a directory. "
                                f"source_dir must be the directory that holds the corpus's {edition}/ folder.{hint}")
    found = sorted(root.glob(fmt.archive_glob[edition]))
    if not found:
        raise FileNotFoundError(f"no {fmt.archive_glob[edition]} files in {root}")
    return found


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")            # the older files; every byte is a character


def _text_tokens(data: bytes) -> List[str]:
    tokens = _decode(data).replace("\x00", " ").split()
    if tokens and tokens[0].startswith("@@"):     # the document's own first token: @@<id>
        del tokens[0]
    return tokens


def _tagged_tokens(data: bytes) -> Tuple[List[str], List[Optional[str]], int, List[str]]:
    """Words and coarse tags of a tagged file. Its first line is the
    document's header. Lines end in CR LF, but the header's often ends in a
    bare LF: lines are split on LF and a trailing CR dropped, or the first
    word of the document would be lost with the header. A word that holds
    spaces is several tokens, as it is in the text edition: nearly always a
    dash and then the word ("- Captain"), so the row's tag goes to the last
    of them and the others have none. A NUL in the word column (the files
    have them where a page mark was) is not a token."""
    words: List[str] = []
    tags: List[Optional[str]] = []
    unreadable = 0
    bad: List[str] = []
    lines = _decode(data).split("\n")
    first = True
    for line in lines:
        line = line.rstrip("\r")
        if not line.strip():
            continue
        cols = line.split("\t")
        if first:
            first = False
            if cols[0].startswith("@@"):
                continue
        if len(cols) < 3:
            unreadable += 1
            if len(bad) < 5:
                bad.append(line[:120])
            continue
        pieces = cols[0].replace("\x00", " ").split()
        if not pieces:
            continue                                  # an empty word, or a NUL where a page mark was
        for piece in pieces[:-1]:
            words.append(piece)
            tags.append(None)
        words.append(pieces[-1])
        tags.append(google_tag(cols[2]))
    return words, tags, unreadable, bad


def documents(archive: str | Path, fmt: CorpusFormat, ngram_type: str) -> Iterator[Document]:
    """Every document of one archive, in the archive's order. A member that
    is not named as the format says raises :class:`ReadError`; a directory
    entry is skipped."""
    tagged = ngram_type == "tagged"
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            base = info.filename.rsplit("/", 1)[-1]
            m = fmt.member.match(base)
            if not m:
                raise ReadError(f"{Path(archive).name}: member {info.filename!r} is not named like a {fmt.label} "
                                f"document ({fmt.member.pattern})")
            data = zf.read(info)
            if tagged:
                words, tags, unreadable, bad = _tagged_tokens(data)
            else:
                words, tags, unreadable, bad = _text_tokens(data), None, 0, []
            # the tagged edition's padding rows, and a first token left over from the file's name
            if fmt.padding and tagged:
                keep = [i for i, w in enumerate(words) if w not in fmt.padding]
                if len(keep) != len(words):
                    words = [words[i] for i in keep]
                    tags = [tags[i] for i in keep]
            if words and words[0] in fmt.leading_junk:
                del words[0]
                if tags is not None:
                    del tags[0]
            yield Document(base, int(m.group("year")), m.group("genre"), words, tags, unreadable, bad,
                           nbytes=info.file_size)


def units(doc: Document, fmt: CorpusFormat, max_tokens: int = 500) -> Iterator[List[str]]:
    """The document's units, each a list of tokens as they will be stored:
    ``word_TAG`` where the token has a tag, the bare word otherwise. A unit
    longer than ``max_tokens`` (text with no sentence punctuation) is cut
    into consecutive pieces of that length."""
    gap, breaks = fmt.gap, fmt.breaks
    tags = doc.tags
    current: List[str] = []
    for i, word in enumerate(doc.tokens):
        if word in SENTENCE_END or word == gap or word.lower() in breaks:
            if current:
                yield current
                current = []
            continue
        tag = tags[i] if tags is not None else None
        current.append(f"{word}_{tag}" if tag else word)
        if len(current) >= max_tokens:
            yield current
            current = []
    if current:
        yield current
