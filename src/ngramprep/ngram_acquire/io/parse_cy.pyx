# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Fast path for the acquisition parser.

``parse_packed(raw, mode, suffixes, bigrams, firsts, case_sensitive)`` does,
in one pass over the raw bytes of a line, what ``parse_line`` +
``_pack_record`` do in Python: strip, split off the n-gram text, combine the
word pairs in ``bigrams`` into hyphenated tokens, apply the
tagged/untagged/all filter, parse the ``year,frequency,document_count``
triples and pack them as little-endian uint64. It returns ``(key, packed)``
or ``None`` (line rejected).

``bigrams`` is a frozenset of UTF-8 ``bytes`` pairs as ``prepare_bigrams``
normalizes them (lowercased unless ``case_sensitive``), or None; ``firsts``
is a 256-byte table with a 1 at the first byte of every entry, so that most
tokens are dismissed by one lookup. A pair that would need Unicode case
folding (a non-ASCII byte in a candidate token when matching ignores case)
is handed back to Python, whose ``str.lower()`` is the reference.

It is exact by construction: whenever a line contains anything whose Python
semantics are not reproduced here (non-ASCII whitespace at the edges, numbers
that ``int()`` would accept but a plain digit scan would not, an undecodable
n-gram, ...), it returns the ``FALLBACK`` sentinel and the caller runs the
original Python code for that line. The differential test in
``tests/test_acquire_parse_cy.py`` checks the two agree on everything else.
"""
from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AS_STRING, PyBytes_GET_SIZE
from cpython.unicode cimport PyUnicode_DecodeUTF8
from libc.stdint cimport uint64_t, uint8_t
from libc.string cimport memcmp

__all__ = ["parse_packed", "bigram_tables", "FALLBACK", "MODE_ALL", "MODE_TAGGED", "MODE_UNTAGGED"]

MODE_ALL = 0
MODE_TAGGED = 1
MODE_UNTAGGED = 2

FALLBACK = object()          # "run the Python parser for this line"

cdef uint64_t UINT64_MAX_DIV10 = 1844674407370955161ULL   # floor(2**64-1 / 10)


def bigram_tables(bigrams):
    """``(frozenset of UTF-8 bytes, firsts)`` for ``parse_packed`` from the
    pairs as ``prepare_bigrams`` returns them; ``(None, None)`` for none."""
    if not bigrams:
        return None, None
    encoded = frozenset(str(b).encode("utf-8") for b in bigrams)
    firsts = bytearray(256)
    for b in encoded:
        if b:
            firsts[b[0]] = 1
    return encoded, bytes(firsts)


cdef inline bint _is_ascii_space(uint8_t c) nogil:
    # str.strip() on ASCII strips exactly these (plus \x1c-\x1f, which are
    # also whitespace for str.strip(); treat those as "unusual" -> fallback).
    return c == 0x20 or c == 0x09 or c == 0x0a or c == 0x0d or c == 0x0b or c == 0x0c


cdef inline bint _is_odd_space(uint8_t c) nogil:
    return c == 0x1c or c == 0x1d or c == 0x1e or c == 0x1f


cdef inline bint _endswith_any(const uint8_t* tok, Py_ssize_t n, list suffixes) except -1:
    cdef bytes s
    cdef Py_ssize_t m
    for s in suffixes:
        m = PyBytes_GET_SIZE(s)
        if m <= n and memcmp(tok + n - m, PyBytes_AS_STRING(s), m) == 0:
            return True
    return False


cdef inline void _put_le64(uint8_t* out, uint64_t v) noexcept nogil:
    cdef int i
    for i in range(8):
        out[i] = <uint8_t>(v & 0xff)
        v >>= 8


cdef int _scan_field(const uint8_t* p, Py_ssize_t n, uint64_t* value) nogil:
    """Parse an all-ASCII-digit field into *value.
    Returns 1 = ok, 0 = int() would fail (skip entry), -1 = unsure (fallback)."""
    cdef Py_ssize_t i
    cdef uint8_t c
    cdef uint64_t v = 0
    if n == 0:
        return 0                       # int("") -> ValueError
    for i in range(n):
        c = p[i]
        if c >= 0x30 and c <= 0x39:
            if v > UINT64_MAX_DIV10 or (v == UINT64_MAX_DIV10 and c - 0x30 > 5):
                return -1              # would not fit uint64: let struct.pack decide
            v = v * 10 + (c - 0x30)
        elif c >= 0x80 or c == 0x5f or c == 0x2b or c == 0x2d or _is_ascii_space(c) or _is_odd_space(c):
            return -1                  # unicode digits, '_', sign, whitespace: int() may accept
        else:
            return 0                   # letters, '.', etc.: int() rejects
    value[0] = v
    return 1


cdef int _passes_filter(const uint8_t* p, Py_ssize_t n, int mode, list sfx) except -1:
    """The tagged/untagged filter over the tokens of p[0:n], split on single
    spaces exactly like str.split(" "). 1 = keep, 0 = reject."""
    cdef Py_ssize_t i, tok_start = 0
    cdef bint any_tagged = False, all_tagged = True
    if mode == 0:
        return 1
    for i in range(n + 1):
        if i == n or p[i] == 0x20:
            if _endswith_any(p + tok_start, i - tok_start, sfx):
                any_tagged = True
            else:
                all_tagged = False
            tok_start = i + 1
    if mode == 1 and not all_tagged:
        return 0
    if mode == 2 and any_tagged:
        return 0
    return 1


cdef inline uint8_t _lower(uint8_t c) noexcept nogil:
    return c + 0x20 if (c >= 0x41 and c <= 0x5a) else c


DEF MAX_TOKENS = 64


cdef object _combine(const uint8_t* p, Py_ssize_t n, frozenset bigrams, const uint8_t* firsts, bint cs):
    """Combine the pairs of ``bigrams`` in the n-gram text p[0:n]: None when
    nothing matched (the text stands), the new text as bytes when something
    did, FALLBACK when Python must judge (Unicode case folding, or a line
    with more tokens than the fixed buffers hold). Mirrors
    ``_combine_bigrams_in_ngram``: tokens split on single spaces, a token's
    base is what precedes its last underscore, the first token's tag
    survives and the second's is dropped."""
    cdef Py_ssize_t ts[MAX_TOKENS]
    cdef Py_ssize_t te[MAX_TOKENS]
    cdef Py_ssize_t ntok = 0, i, j, s1, e1, s2, e2, u1, u2, m, k
    cdef uint8_t c
    cdef bint matched = False, nonascii
    cdef bytes cand
    cdef uint8_t* cb
    cdef list parts

    # token spans, as str.split(" ") makes them (empty tokens included)
    ts[0] = 0
    for i in range(n + 1):
        if i == n or p[i] == 0x20:
            te[ntok] = i
            ntok += 1
            if i < n:
                if ntok == MAX_TOKENS:
                    return FALLBACK
                ts[ntok] = i + 1
    if ntok < 2:
        return None

    parts = []
    i = 0
    while i < ntok:
        if i + 1 < ntok:
            s1 = ts[i]; e1 = te[i]; s2 = ts[i + 1]; e2 = te[i + 1]
            # the bases: up to the last underscore, when there is one
            u1 = e1
            for j in range(e1 - 1, s1 - 1, -1):
                if p[j] == 0x5f:
                    u1 = j
                    break
            u2 = e2
            for j in range(e2 - 1, s2 - 1, -1):
                if p[j] == 0x5f:
                    u2 = j
                    break
            # one lookup dismisses most tokens: the candidate's first byte
            c = p[s1] if u1 > s1 else 0x20
            if not cs:
                c = _lower(c)
            if firsts[c]:
                m = (u1 - s1) + 1 + (u2 - s2)
                cand = PyBytes_FromStringAndSize(NULL, m)
                cb = <uint8_t*> PyBytes_AS_STRING(cand)
                k = 0
                nonascii = False
                for j in range(s1, u1):
                    c = p[j]
                    if c >= 0x80:
                        nonascii = True
                    cb[k] = c if cs else _lower(c)
                    k += 1
                cb[k] = 0x20
                k += 1
                for j in range(s2, u2):
                    c = p[j]
                    if c >= 0x80:
                        nonascii = True
                    cb[k] = c if cs else _lower(c)
                    k += 1
                if nonascii and not cs:
                    return FALLBACK           # str.lower() may fold what ASCII lowering cannot
                if cand in bigrams:
                    matched = True
                    parts.append(PyBytes_FromStringAndSize(<const char*> (p + s1), u1 - s1) + b"-"
                                 + PyBytes_FromStringAndSize(<const char*> (p + s2), u2 - s2)
                                 + PyBytes_FromStringAndSize(<const char*> (p + u1), e1 - u1))
                    i += 2
                    continue
        parts.append(PyBytes_FromStringAndSize(<const char*> (p + ts[i]), te[i] - ts[i]))
        i += 1
    if not matched:
        return None
    return b" ".join(parts)


def parse_packed(bytes raw, int mode, tuple suffixes, frozenset bigrams=None, bytes firsts=None,
                 bint case_sensitive=False):
    """See module docstring. ``suffixes`` are the tag suffixes as bytes
    (``b"_NOUN"``, ...); ignored for ``MODE_ALL``. ``bigrams`` and
    ``firsts`` come from ``bigram_tables``; both None for no combining."""
    cdef const uint8_t* buf = <const uint8_t*> PyBytes_AS_STRING(raw)
    cdef Py_ssize_t n = PyBytes_GET_SIZE(raw)
    cdef Py_ssize_t start = 0, end = n, i, tab
    cdef uint8_t c
    cdef list sfx = list(suffixes)
    cdef object combined
    cdef bytes text
    cdef const uint8_t* tbuf
    cdef Py_ssize_t tlen

    # --- strip (ASCII whitespace); anything unusual at the edges -> fallback
    while start < end and _is_ascii_space(buf[start]):
        start += 1
    while end > start and _is_ascii_space(buf[end - 1]):
        end -= 1
    if start >= end:
        return None
    if buf[end - 1] >= 0x80 or _is_odd_space(buf[start]) or _is_odd_space(buf[end - 1]):
        return FALLBACK

    # --- n-gram text up to the first tab
    tab = -1
    for i in range(start, end):
        if buf[i] == 0x09:
            tab = i
            break
    if tab < 0:
        return None                    # no frequency data
    # (tab == start cannot happen: a leading tab was stripped)

    try:
        key = PyUnicode_DecodeUTF8(<const char*> (buf + start), tab - start, NULL)
    except UnicodeDecodeError:
        return FALLBACK                # the Python path logs this the usual way
    if buf[start] >= 0x80 and key[0].isspace():
        return FALLBACK                # str.strip() would have removed it

    # --- combine word pairs, then filter the result (the Python order)
    tbuf = buf + start
    tlen = tab - start
    if bigrams is not None:
        combined = _combine(tbuf, tlen, bigrams, <const uint8_t*> PyBytes_AS_STRING(firsts), case_sensitive)
        if combined is FALLBACK:
            return FALLBACK
        if combined is not None:
            text = <bytes> combined
            tbuf = <const uint8_t*> PyBytes_AS_STRING(text)
            tlen = PyBytes_GET_SIZE(text)
            key = PyUnicode_DecodeUTF8(<const char*> tbuf, tlen, NULL)   # slices of a valid string: decodes
    if not _passes_filter(tbuf, tlen, mode, sfx):
        return None

    # --- frequency triples
    cdef Py_ssize_t fs = tab + 1, p, q, f0, f1, f2, nfields, ntrip = 0, cap = 8
    cdef uint64_t vals[3]
    cdef int rc
    cdef bytes packed = PyBytes_FromStringAndSize(NULL, cap * 24)
    cdef uint8_t* out = <uint8_t*> PyBytes_AS_STRING(packed)
    cdef Py_ssize_t bounds[4]

    for i in range(fs, end):
        if buf[i] >= 0x80:
            return FALLBACK            # non-ASCII in the numbers: let Python judge
    p = fs
    while p <= end:
        # entry = buf[p:q), q = next tab or end
        q = p
        while q < end and buf[q] != 0x09:
            q += 1
        # split entry on commas
        nfields = 0
        bounds[0] = p
        for i in range(p, q):
            if buf[i] == 0x2c:
                nfields += 1
                if nfields <= 3:
                    bounds[nfields] = i
        if nfields == 2:
            f0 = bounds[0]; f1 = bounds[1]; f2 = bounds[2]
            rc = _scan_field(buf + f0, f1 - f0, &vals[0])
            if rc == 1:
                rc = _scan_field(buf + f1 + 1, f2 - f1 - 1, &vals[1])
            if rc == 1:
                rc = _scan_field(buf + f2 + 1, q - f2 - 1, &vals[2])
            if rc < 0:
                return FALLBACK
            if rc == 1:
                if ntrip == cap:
                    cap *= 2
                    packed = packed[: ntrip * 24] + PyBytes_FromStringAndSize(NULL, (cap - ntrip) * 24)
                    out = <uint8_t*> PyBytes_AS_STRING(packed)
                _put_le64(out + ntrip * 24, vals[0])
                _put_le64(out + ntrip * 24 + 8, vals[1])
                _put_le64(out + ntrip * 24 + 16, vals[2])
                ntrip += 1
        p = q + 1

    if ntrip == 0:
        return None

    return key, packed[: ntrip * 24]
