#!/usr/bin/env python3
"""Rebuild gensim's ``corpus_file`` reader so it runs without the GIL.

gensim declares the line reader of its multi-threaded ``corpus_file`` training
path as ``cpdef ... nogil except *``. With the Cython that gensim builds with
(< 3.0), a ``nogil except *`` function re-acquires the GIL after *every call*
to check for an exception, and ``read_sentence`` is called once per line. On
ordinary text (dozens of words a line) that is a small tax; on a corpus of
n-grams (five tokens a line) the GIL handshake costs more than the training
of the line, every thread of a model fights for the same lock, and a model
tops out at three or four cores however many threads it is given.

This script edits the installed gensim's ``word2vec_corpusfile.pxd``/``.pyx``
(the wheel ships them): the three declarations lose their ``except *`` (the
C++ reader only throws on allocation failure, which Cython then reports as an
unraisable error rather than propagate). It then recompiles that one
extension module in place, with the Cython on the path (which must be
< 3.0, as gensim's own build requires) and the installed numpy's headers.
The other ``*_corpusfile`` modules keep their compiled form: the change does
not alter the methods' signatures, only the caller's post-call check.

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

DECLARATIONS = (
    "cpdef vector[string] read_sentence(self) nogil",
    "cpdef vector[vector[string]] _read_chunked_sentence(self) nogil",
    "cpdef vector[vector[string]] next_batch(self) nogil",
)
MODULE = "gensim.models.word2vec_corpusfile"


def edit_sources(root: Path) -> None:
    """Drop ``except *`` from the reader's declarations under ``root`` (the
    directory holding ``gensim/``). Fails loudly on an unexpected gensim."""
    for rel, suffix in (("gensim/models/word2vec_corpusfile.pxd", ""),
                        ("gensim/models/word2vec_corpusfile.pyx", ":")):
        path = root / rel
        text = path.read_text()
        for decl in DECLARATIONS:
            old = f"{decl} except *{suffix}"
            n = text.count(old)
            if n != 1:
                sys.exit(f"{path}: expected one occurrence of {old!r}, found {n}; "
                         "is this gensim 4.3.x, and not already patched?")
            text = text.replace(old, f"{decl}{suffix}")
        path.write_text(text)
        print(f"patched {path}")


def rebuild_installed() -> None:
    import Cython
    if int(Cython.__version__.split(".")[0]) >= 3:
        sys.exit(f"Cython {Cython.__version__} on the path: gensim must be built with Cython < 3.0 "
                 "(Cython 3 checks for exceptions after every cdef call, which is the slowdown "
                 "this patch removes)")
    import numpy
    from Cython.Build import cythonize
    from setuptools import Distribution, Extension
    import gensim.models
    models_dir = Path(gensim.models.__file__).parent
    root = models_dir.parent.parent
    for name in ("word2vec_corpusfile.pyx", "word2vec_corpusfile.pxd", "word2vec_inner.pxd",
                 "fast_line_sentence.h", "voidptr.h"):
        if not (models_dir / name).exists():
            sys.exit(f"{models_dir / name} missing: this gensim installation does not ship its sources")
    edit_sources(root)
    ext = Extension(MODULE, sources=[str(models_dir / "word2vec_corpusfile.pyx")], language="c++",
                    include_dirs=[numpy.get_include(), str(models_dir)],
                    extra_compile_args=["-std=c++11", "-O3"], extra_link_args=["-std=c++11"])
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(root)                     # so Cython resolves gensim.models.* cimports from the package
        try:
            dist = Distribution({"name": "gensim-corpusfile-nogil",
                                 "ext_modules": cythonize([ext], language_level=3, quiet=True)})
            cmd = dist.get_command_obj("build_ext")
            cmd.build_lib = tmp
            cmd.build_temp = str(Path(tmp) / "obj")
            cmd.ensure_finalized()
            cmd.run()
        finally:
            os.chdir(cwd)
        built = list((Path(tmp) / "gensim" / "models").glob("word2vec_corpusfile*.so"))
        if len(built) != 1:
            sys.exit(f"expected one built module, found {built}")
        for old in models_dir.glob("word2vec_corpusfile*.so"):
            old.unlink()
        target = models_dir / built[0].name
        shutil.copy2(built[0], target)
        print(f"rebuilt {target}")
    # the module must import, and the reader must still read
    import importlib
    mod = importlib.import_module(MODULE)
    sample = Path(tempfile.mkdtemp()) / "s.txt"
    sample.write_text("a b c\nd e\n")
    assert [list(s) for s in mod.CythonLineSentence(str(sample))] == [[b"a", b"b", b"c"], [b"d", b"e"]]
    print("gensim corpus_file reader rebuilt without the GIL")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--edit":
        edit_sources(Path(sys.argv[2]))
    elif len(sys.argv) == 1:
        rebuild_installed()
    else:
        sys.exit(__doc__)
