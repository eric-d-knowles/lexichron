"""lexichron.training: grid, corpus files, stability and benchmarks (no gensim needed)."""
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import fake_rocks                                      # noqa: E402
from ngramprep.common_db import api as dbapi            # noqa: E402
from lexichron.training import benchmarks, corpus, grid, stability, trainer   # noqa: E402


@pytest.fixture
def rocks(monkeypatch):
    fake_rocks.reset()
    monkeypatch.setattr(dbapi, "rs", fake_rocks)
    return fake_rocks


# ------------------------------------------------------------------ grid
def test_grid_parses_lists_bare_values_and_strings():
    g = grid.Grid.from_settings({"vector_size": [100, 300], "window": "2, 5", "approach": "CBOW", "epochs": 5})
    assert g.values["vector_size"] == (100, 300) and g.values["window"] == (2, 5)
    assert g.values["approach"] == ("cbow",) and g.values["epochs"] == (5,)
    assert g.values["min_count"] == (5,) and g.values["weight_by"] == ("freq",)      # defaults
    assert len(g) == 4 and len(g.cells()) == 4
    assert g.describe() == "2 vector sizes x 2 windows x 1 min count x 1 epochs x 1 approach x 1 weighting = 4 cells"
    c = g.cells()[0]
    assert c.stem == "vs100_w2_mc5_e5_sg0_wbfreq" and grid.Cell.from_stem(c.stem) == c
    assert grid.corpus_variants(g.cells()) == ["freq"]
    with pytest.raises(ValueError):
        grid.Grid.from_settings({"vector_size": "100, ten"})
    with pytest.raises(ValueError):
        grid.Grid.from_settings({"approach": ["glove"]})
    with pytest.raises(ValueError):
        grid.Grid.from_settings({"window": []})


# ------------------------------------------------------------------ corpus
def _pivoted(path, rows):
    db = fake_rocks.open(str(path), mode="rw", create_if_missing=True)
    with db.write_batch() as wb:
        for year, ng, count, vol in rows:
            wb.put(struct.pack(">I", year) + ng, struct.pack("<QQQ", year, count, vol))
    return db


def test_corpus_file_weights_strips_and_splits(rocks, tmp_path):
    db = tmp_path / "5grams_pivoted.db"
    _pivoted(db, [(1990, b"the cat sat on <UNK>", 4, 2),          # weight floor(log5(5)) = 1
                  (1990, b"<UNK> <UNK> <UNK> <UNK> dog", 1000, 1),   # one real token: stripped away
                  (1990, b"a big dog ran <UNK>", 124, 124),        # weight floor(log5(125)) = 3
                  (1991, b"next year only", 10, 10)])
    out = tmp_path / "c.txt"
    st = corpus.write_corpus(str(db), 1990, str(out), halves=(str(tmp_path / "a.txt"), str(tmp_path / "b.txt")))
    lines = out.read_text().splitlines()
    assert st.ngrams_read == 3 and st.ngrams_used == 2
    assert lines == ["a big dog ran"] * 3 + ["the cat sat on"] and st.lines == 4 and st.tokens == 16   # key order
    a = (tmp_path / "a.txt").read_text().splitlines()
    b = (tmp_path / "b.txt").read_text().splitlines()
    assert sorted(a + b) == sorted(lines) and st.lines_half_a + st.lines_half_b == 4
    # reject drops any n-gram with <UNK>; retain keeps the token
    st = corpus.write_corpus(str(db), 1990, str(out), unk_mode="reject")
    assert st.ngrams_used == 0 and out.read_text() == ""
    st = corpus.write_corpus(str(db), 1990, str(out), unk_mode="retain", weight_by="none")
    assert st.lines == 3 and "<UNK>" in out.read_text()
    # a Poisson resample changes the repeat counts but is reproducible
    s1 = corpus.write_corpus(str(db), 1990, str(out), resample_seed=7)
    s2 = corpus.write_corpus(str(db), 1990, str(tmp_path / "d.txt"), resample_seed=7)
    assert s1.lines == s2.lines
    assert not list(tmp_path.glob("*.part"))
    assert corpus.corpus_file_name(1990, "freq", "strip", "a") == "corpus_y1990_wbfreq_strip_a.txt"


# ------------------------------------------------------------------ vectors, stability, benchmarks
def _fake_vectors(words, dim=8, seed=0, noise=0.0, base=None):
    rng = np.random.default_rng(seed)
    m = base.copy() if base is not None else rng.normal(size=(len(words), dim)).astype(np.float32)
    if noise:
        m = m + noise * rng.normal(size=m.shape).astype(np.float32)
    return trainer.Vectors(words=list(words), matrix=m, counts=np.arange(len(words), 0, -1))


def test_vectors_round_trip_and_probe_sampling(tmp_path):
    words = [f"w{i:03d}" for i in range(200)]
    v = _fake_vectors(words)
    trainer.save_vectors(v, str(tmp_path / "m.npz"))
    back = trainer.load_vectors(str(tmp_path / "m.npz"))
    assert back.words == words and np.allclose(back.matrix, v.matrix) and back.counts[0] == 200
    other = _fake_vectors(words[:150])                   # a smaller vocabulary
    probes = stability.probe_words([v, other], per_decile=3, seed=1)
    assert len(probes) == 30 and all(p in other for p in probes)
    assert probes == stability.probe_words([v, other], per_decile=3, seed=1)   # same seed, same probes


def test_neighbour_overlap_tracks_agreement():
    words = [f"w{i:03d}" for i in range(300)]
    a = _fake_vectors(words, dim=16, seed=1)
    same = _fake_vectors(words, dim=16, base=a.matrix)
    noisy = _fake_vectors(words, dim=16, seed=2, noise=0.3, base=a.matrix)
    unrelated = _fake_vectors(words, dim=16, seed=3)
    probes = stability.probe_words([a], per_decile=10)
    full = stability.neighbour_overlap(a, same, probes)
    mid = stability.neighbour_overlap(a, noisy, probes)
    none = stability.neighbour_overlap(a, unrelated, probes)
    assert full == 1.0 and 0.2 < mid < 1.0 and none < 0.15
    assert stability.aligned_cosine(a, same, probes) > 0.999
    assert stability.aligned_cosine(a, noisy, probes) < stability.aligned_cosine(a, same, probes)
    # a rotated copy is the same model geometrically: overlap and aligned cosine both see that
    q, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(16, 16)))
    rotated = trainer.Vectors(words=words, matrix=(a.matrix @ q).astype(np.float32))
    assert stability.neighbour_overlap(a, rotated, probes) == 1.0
    assert stability.aligned_cosine(a, rotated, probes) > 0.999


def test_benchmarks_on_a_constructed_model(tmp_path):
    words = ["king", "queen", "man", "woman", "cat", "dog", "car", "bus"]
    m = np.zeros((8, 3), dtype=np.float32)
    m[0] = [1, 1, 0]; m[1] = [1, -1, 0]; m[2] = [0, 1, 0]; m[3] = [0, -1, 0]      # king - man + woman = queen
    m[4] = [0, 0, 1]; m[5] = [0.1, 0, 1]; m[6] = [0, 0, -1]; m[7] = [0.1, 0, -1]
    v = trainer.Vectors(words=words, matrix=m)
    sim = tmp_path / "sim.tsv"
    sim.write_text("# Word 1\tWord 2\tHuman (mean)\ncat\tdog\t9.0\ncar\tbus\t8.5\ncat\tcar\t1.0\ndog\tbus\t1.5\nking\tzebra\t5\n")
    rho, cov = benchmarks.similarity_score(v, str(sim))
    assert rho > 0.9 and cov == 0.8                         # zebra is not in the model
    ana = tmp_path / "ana.txt"
    ana.write_text(": family\nking queen man woman\nman woman king queen\ncat dog zebra fish\n")
    acc, cov = benchmarks.analogy_accuracy(v, str(ana))
    assert acc == 1.0 and abs(cov - 2 / 3) < 1e-9
    assert benchmarks.coverage(v, ["king", "cat", "zebra", "fish"]) == 0.5


def test_simlex_format_with_a_pos_column_is_read(tmp_path):
    words = ["old", "new", "smart", "intelligent", "hard", "difficult"]
    m = np.array([[1, 0], [0, 1], [0, 1], [0, 1], [1, 1], [1, 0.9]], dtype=np.float32)
    v = trainer.Vectors(words=words, matrix=m)
    f = tmp_path / "simlex999.txt"
    f.write_text("word1\tword2\tPOS\tSimLex999\tconc(w1)\nold\tnew\tA\t1.58\t2.72\nsmart\tintelligent\tA\t9.2\t1.75\nhard\tdifficult\tA\t8.77\t3.76\n")
    rho, cov = benchmarks.similarity_score(v, str(f))
    assert cov == 1.0 and rho > 0.9


# ------------------------------------------------------------------ the gensim patch for the image
def test_patch_gensim_edits_the_corpus_file_reader(tmp_path):
    import subprocess
    models = tmp_path / "gensim" / "models"
    models.mkdir(parents=True)
    pxd = ("cdef class CythonLineSentence:\n"
           "    cpdef bool_t is_eof(self) nogil\n"
           "    cpdef vector[string] read_sentence(self) except * nogil\n"
           "    cpdef vector[vector[string]] _read_chunked_sentence(self) except * nogil\n"
           "    cpdef vector[vector[string]] _chunk_sentence(self, vector[string] sent) nogil\n"
           "    cpdef void reset(self) noexcept nogil\n"
           "    cpdef vector[vector[string]] next_batch(self) except * nogil\n"
           "cdef class CythonVocab:\n"
           "    cdef cvocab_t* get_vocab_ptr(self) except * nogil\n")
    pyx = "".join(line + ":\n    pass\n" for line in pxd.splitlines() if line.startswith("    "))
    (models / "word2vec_corpusfile.pxd").write_text(pxd)
    (models / "word2vec_corpusfile.pyx").write_text(pyx)
    inner = ("ctypedef float (*sdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) except -1 nogil\n"
             "ctypedef double (*dsdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) except -1 nogil\n"
             "ctypedef void (*sscal_ptr) (const int *N, const float *alpha, const float *X, const int *incX) noexcept nogil\n")
    (models / "word2vec_inner.pxd").write_text(inner)
    script = Path(__file__).parent.parent / "scripts" / "patch_gensim.py"
    subprocess.run([sys.executable, str(script), "--edit", str(tmp_path)], check=True, capture_output=True)
    out_pxd, out_pyx = (models / "word2vec_corpusfile.pxd").read_text(), (models / "word2vec_corpusfile.pyx").read_text()
    assert "except *" not in out_pxd and "except *" not in out_pyx
    out_inner = (models / "word2vec_inner.pxd").read_text()
    assert "except -1" not in out_inner and out_inner.count("noexcept nogil") == 3   # the dot pointers, like sscal
    for line in out_pxd.splitlines():
        if "nogil" in line:
            assert "noexcept nogil" in line, line                 # every reader method is now noexcept
    assert "cpdef vector[string] read_sentence(self) noexcept nogil:" in out_pyx
    assert "cpdef void reset(self) noexcept nogil" in out_pxd       # already fine: untouched
    # a second run (or another gensim) fails loudly rather than silently doing nothing
    r = subprocess.run([sys.executable, str(script), "--edit", str(tmp_path)], capture_output=True, text=True)
    assert r.returncode != 0 and "expected one occurrence" in r.stderr


def test_cell_stem_round_trips_a_value_with_an_underscore():
    c = grid.Cell(vector_size=100, window=5, min_count=1, epochs=10, approach="skip-gram", weight_by="doc_freq")
    assert c.stem == "vs100_w5_mc1_e10_sg1_wbdoc_freq" and grid.Cell.from_stem(c.stem) == c
    assert grid.Cell.from_stem("vs300_w2_mc5_e5_sg0_wbnone").approach == "cbow"
    with pytest.raises(ValueError):
        grid.Cell.from_stem("y1990_vs100_w5_mc1_e10_sg1_wbnone_s0")
