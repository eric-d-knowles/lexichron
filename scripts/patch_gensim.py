#!/usr/bin/env python3
"""Patch a gensim source tree so its ``corpus_file`` reader runs without the GIL.

gensim declares the line reader of its multi-threaded ``corpus_file`` training
path as ``cpdef ... nogil except *``. With the Cython that gensim builds with
(< 3.0), a ``nogil except *`` function re-acquires the GIL after *every call*
to check for an exception, and ``read_sentence`` is called once per line. On
ordinary text (dozens of words a line) that is a small tax; on a corpus of
n-grams (five tokens a line) the GIL handshake costs more than the training
of the line, every thread of a model fights for the same lock, and a model
tops out at three or four cores however many threads it is given.

The three declarations lose their ``except *`` (the C++ reader only throws on
allocation failure, which Cython would then report as an unraisable error
rather than propagate). The generated C++ of the three ``*_corpusfile``
modules is removed so that gensim's setup.py regenerates them from the
patched sources.

Usage: patch_gensim.py <unpacked gensim sdist directory>
"""
import sys
from pathlib import Path

DECLARATIONS = (
    "cpdef vector[string] read_sentence(self) nogil",
    "cpdef vector[vector[string]] _read_chunked_sentence(self) nogil",
    "cpdef vector[vector[string]] next_batch(self) nogil",
)


def patch(root: Path) -> None:
    for rel, suffix in (("gensim/models/word2vec_corpusfile.pxd", ""),
                        ("gensim/models/word2vec_corpusfile.pyx", ":")):
        path = root / rel
        text = path.read_text()
        for decl in DECLARATIONS:
            old = f"{decl} except *{suffix}"
            new = f"{decl}{suffix}"
            n = text.count(old)
            if n != 1:
                sys.exit(f"{path}: expected one occurrence of {old!r}, found {n}; "
                         "is this gensim 4.3.x? Adjust the patch for this version.")
            text = text.replace(old, new)
        path.write_text(text)
        print(f"patched {path}")
    for name in ("word2vec", "doc2vec", "fasttext"):
        cpp = root / "gensim" / "models" / f"{name}_corpusfile.cpp"
        if cpp.exists():
            cpp.unlink()
            print(f"removed {cpp} (regenerated from the .pyx at build)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    patch(Path(sys.argv[1]))
