# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Fast path for the acquisition parser.

``parse_packed(raw, mode, suffixes)`` does, in one pass over the raw bytes of
a line, what ``parse_line`` + ``_pack_record`` do in Python: strip, split off
the n-gram text, apply the tagged/untagged/all filter, parse the
``year,frequency,document_count`` triples and pack them as little-endian
uint64. It returns ``(key, packed)`` or ``None`` (line rejected).

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

__all__ = ["parse_packed", "FALLBACK", "MODE_ALL", "MODE_TAGGED", "MODE_UNTAGGED"]

MODE_ALL = 0
MODE_TAGGED = 1
MODE_UNTAGGED = 2

FALLBACK = object()          # "run the Python parser for this line"

cdef uint64_t UINT64_MAX_DIV10 = 1844674407370955161ULL   # floor(2**64-1 / 10)


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


def parse_packed(bytes raw, int mode, tuple suffixes):
    """See module docstring. ``suffixes`` are the tag suffixes as bytes
    (``b"_NOUN"``, ...); ignored for ``MODE_ALL``."""
    cdef const uint8_t* buf = <const uint8_t*> PyBytes_AS_STRING(raw)
    cdef Py_ssize_t n = PyBytes_GET_SIZE(raw)
    cdef Py_ssize_t start = 0, end = n, i, tab, tok_start, ntok
    cdef uint8_t c
    cdef bint any_tagged = False, all_tagged = True
    cdef list sfx = list(suffixes)

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

    # --- filter: tokens are split on single spaces, exactly like str.split(" ")
    if mode != 0:
        tok_start = start
        ntok = 0
        for i in range(start, tab + 1):
            if i == tab or buf[i] == 0x20:
                ntok += 1
                if _endswith_any(buf + tok_start, i - tok_start, sfx):
                    any_tagged = True
                else:
                    all_tagged = False
                tok_start = i + 1
        if mode == 1 and not all_tagged:
            return None
        if mode == 2 and any_tagged:
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
