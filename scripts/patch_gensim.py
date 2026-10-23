#!/usr/bin/env python3
"""Rebuild gensim's ``corpus_file`` reader so it runs without the GIL.

gensim's multi-threaded ``corpus_file`` training path reads lines through
Cython methods declared ``nogil`` but not ``noexcept`` (``read_sentence`` is
``except * nogil``, ``_chunk_sentence`` has no exception clause at all). With
Cython 3, which gensim 4.4 builds with, a ``nogil`` function that may raise
must re-acquire the GIL after *every call* to check whether it did, and
those methods are called two or three times per line. On ordinary text
(dozens of words a line) that is a small tax; on a corpus of n-grams (five
tokens a line) the GIL handshakes cost more than the training of the line,
every thread of a model fights for the same lock, and a model tops out at
three or four cores however many threads it is given.

This script edits the installed gensim's ``word2vec_corpusfile.pxd``/``.pyx``
(the wheel ships them), declaring the reader's methods ``noexcept`` (the C++
reader only throws on allocation failure, which Cython then reports as an
unraisable error rather than propagate), and ``word2vec_inner.pxd``, whose
BLAS dot pointers otherwise take a dot product of exactly -1 for an error
and print "Exception ignored in our_dot_float". It then recompiles, in place, the
six extension modules of the word2vec/doc2vec/fastText family: they share
C-level symbols whose signatures are checked at import, and those signature
strings differ between Cython releases, so the family must be compiled by
one Cython (gensim 4.4 builds with >= 3.1.3; pin one on the path). It uses
the installed numpy's headers, and checks in a fresh interpreter that gensim
still imports whole and the reader still reads.

Usage:
    patch_gensim.py            patch and rebuild the installed gensim
    patch_gensim.py --edit DIR only edit the sources under DIR (tests)
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

# (declaration as gensim 4.4 writes it, declaration freed of the exception check)
EDITS = (
    ("cpdef bool_t is_eof(self) nogil",
     "cpdef bool_t is_eof(self) noexcept nogil"),
    ("cpdef vector[string] read_sentence(self) except * nogil",
     "cpdef vector[string] read_sentence(self) noexcept nogil"),
    ("cpdef vector[vector[string]] _read_chunked_sentence(self) except * nogil",
     "cpdef vector[vector[string]] _read_chunked_sentence(self) noexcept nogil"),
    ("cpdef vector[vector[string]] _chunk_sentence(self, vector[string] sent) nogil",
     "cpdef vector[vector[string]] _chunk_sentence(self, vector[string] sent) noexcept nogil"),
    ("cpdef vector[vector[string]] next_batch(self) except * nogil",
     "cpdef vector[vector[string]] next_batch(self) noexcept nogil"),
    ("cdef cvocab_t* get_vocab_ptr(self) except * nogil",
     "cdef cvocab_t* get_vocab_ptr(self) noexcept nogil"),
)
# gensim declares the BLAS dot pointers with ``except -1``, so a dot product
# of exactly -1 is taken for an error signal: the value is still returned and
# used, but Cython reports "Exception ignored in our_dot_float" for each one.
# Declaring the pointers noexcept, as the other BLAS pointers are, ends that.
INNER_EDITS = (
    ("ctypedef float (*sdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) except -1 nogil",
     "ctypedef float (*sdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) noexcept nogil"),
    ("ctypedef double (*dsdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) except -1 nogil",
     "ctypedef double (*dsdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) noexcept nogil"),
)
# module -> language; every module that cimports gensim.models.word2vec_inner, plus itself
MODULES = {"word2vec_inner": "c", "word2vec_corpusfile": "c++", "doc2vec_inner": "c++",
           "doc2vec_corpusfile": "c++", "fasttext_inner": "c", "fasttext_corpusfile": "c++"}


def edit_sources(root: Path) -> None:
    """Apply EDITS to the reader's pxd and pyx, and INNER_EDITS to the BLAS
    declarations, under ``root`` (the directory holding ``gensim/``). Fails
    loudly on an unexpected gensim."""
    for rel, suffix, edits in (("gensim/models/word2vec_corpusfile.pxd", "", EDITS),
                               ("gensim/models/word2vec_corpusfile.pyx", ":", EDITS),
                               ("gensim/models/word2vec_inner.pxd", "", INNER_EDITS)):
        path = root / rel
        text = path.read_text()
        for old, new in edits:
            n = text.count(old + suffix)
            if n != 1:
                sys.exit(f"{path}: expected one occurrence of {old + suffix!r}, found {n}; "
                         "is this gensim 4.4.x, and not already patched?")
            text = text.replace(old + suffix, new + suffix)
        path.write_text(text)
        print(f"patched {path}")


def rebuild_installed() -> None:
    import Cython
    if int(Cython.__version__.split(".")[0]) < 3:
        sys.exit(f"Cython {Cython.__version__} on the path: gensim 4.4 builds with Cython >= 3.1")
    import numpy
    from Cython.Build import cythonize
    from setuptools import Distribution, Extension
    import gensim.models
    models_dir = Path(gensim.models.__file__).parent
    root = models_dir.parent.parent
    missing = [f"{m}.pyx" for m in MODULES if not (models_dir / f"{m}.pyx").exists()]
    missing += [n for n in ("word2vec_corpusfile.pxd", "word2vec_inner.pxd", "doc2vec_inner.pxd",
                            "fasttext_inner.pxd", "fast_line_sentence.h", "voidptr.h")
                if not (models_dir / n).exists()]
    if missing:
        sys.exit(f"{models_dir}: {missing} missing; this gensim installation does not ship its sources")
    edit_sources(root)
    exts = []
    for m, lang in MODULES.items():
        std = ["-std=c++11"] if lang == "c++" else []
        exts.append(Extension(f"gensim.models.{m}", sources=[str(models_dir / f"{m}.pyx")], language=lang,
                              include_dirs=[numpy.get_include(), str(models_dir)],
                              extra_compile_args=std + ["-O3"], extra_link_args=std))
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(root)                     # so Cython resolves gensim.models.* cimports from the package
        try:
            dist = Distribution({"name": "gensim-corpusfile-nogil",
                                 "ext_modules": cythonize(exts, language_level=3, quiet=True)})
            cmd = dist.get_command_obj("build_ext")
            cmd.build_lib = tmp
            cmd.build_temp = str(Path(tmp) / "obj")
            cmd.ensure_finalized()
            cmd.run()
        finally:
            os.chdir(cwd)
        for m in MODULES:
            built = list((Path(tmp) / "gensim" / "models").glob(f"{m}.*.so"))
            if len(built) != 1:
                sys.exit(f"expected one built module for {m}, found {built}")
            for old in models_dir.glob(f"{m}.*.so"):
                old.unlink()
            shutil.copy2(built[0], models_dir / built[0].name)
            print(f"rebuilt {models_dir / built[0].name}")
    # gensim must still import whole, and the reader must still read: checked in
    # a fresh interpreter, since this one loaded the old modules before the rebuild
    import subprocess
    sample = Path(tempfile.mkdtemp()) / "s.txt"
    sample.write_text("a b c\nd e\n")
    check = ("import gensim, gensim.models.word2vec, gensim.models.doc2vec, gensim.models.fasttext\n"
             "from gensim.models.word2vec_corpusfile import CythonLineSentence\n"
             f"got = [list(s) for s in CythonLineSentence({str(sample)!r})]\n"
             "assert got == [[b'a', b'b', b'c'], [b'd', b'e']], got\n"
             "print('gensim corpus_file reader rebuilt without the GIL')\n")
    subprocess.run([sys.executable, "-c", check], check=True, cwd=tempfile.gettempdir())


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--edit":
        edit_sources(Path(sys.argv[2]))
    elif len(sys.argv) == 1:
        rebuild_installed()
    else:
        sys.exit(__doc__)
