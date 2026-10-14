"""Look inside a database or vocabulary file: a few records from the start,
from a key, or with a prefix, decoded for the layout the file uses.

This backs ``lexichron inspect`` and the UI's Inspect tab. Databases are
opened read-only with the scan profile, which costs the same whatever
their size, so a login node is fine. A key layout is sniffed from the first
records unless one is given:

- ``pivoted``: 4-byte big-endian year then the n-gram (what the filter stage
  writes for 2-5-grams); values are (year, count, volumes).
- ``ngram``: the n-gram, tagged or not; values are packed year series.
- ``davies``: 4-byte genre, 8-byte year, text (Davies corpora).
- ``raw``: anything else, shown as bytes.
- ``text``: a vocabulary file, one token per line in frequency order.
"""
from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

LAYOUTS = ("auto", "pivoted", "ngram", "davies", "raw")
MODES = ("head", "key", "prefix")
BOOKKEEPING_PREFIXES = (b"__rangemap__/", b"__processed__/")

Row = Tuple[str, str]      # (key as shown, value as shown)


# ---------------------------------------------------------------- what is it
def describe(path: str) -> Dict[str, Any]:
    """What the path holds: ``kind`` (rocksdb, text, missing), and for a
    database ``records`` (data keys, excluding the stages' bookkeeping),
    ``bytes`` on disk, ``compacted`` and the sniffed ``layout``; for a text
    file ``lines`` and ``bytes``. Errors are reported in ``error``."""
    p = Path(path).expanduser()
    if not p.exists():
        return {"kind": "missing", "error": f"{p}: no such file or directory"}
    if p.is_file():
        try:
            with open(p, "rb") as fh:
                lines = sum(1 for _ in fh)
        except OSError as exc:
            return {"kind": "text", "error": str(exc)}
        return {"kind": "text", "lines": lines, "bytes": p.stat().st_size, "layout": "text"}
    try:
        with _open(p) as db:
            records = _int_property(db, "rocksdb.estimate-num-keys")
            size = _int_property(db, "rocksdb.total-sst-files-size")
            level0 = _int_property(db, "rocksdb.num-files-at-level0")
            bookkeeping = sum(1 for _ in _bookkeeping_keys(db))
            layout = sniff_layout(db)
    except Exception as exc:
        return {"kind": "rocksdb", "error": f"cannot open as a RocksDB database: {exc}"}
    from lexichron.provenance import read_provenance
    return {"kind": "rocksdb", "records": max(records - bookkeeping, 0), "bytes": size,
            "compacted": level0 == 0, "layout": layout, "bookkeeping": bookkeeping,
            "provenance": read_provenance(p)}


def sniff_layout(db) -> str:
    """Guess the key layout from the first data records."""
    keys = []
    for k, v in _scan(db, b""):
        if _is_bookkeeping(k):
            continue
        keys.append((bytes(k), bytes(v)))
        if len(keys) >= 8:
            break
    if not keys:
        return "raw"
    if all(len(k) >= 12 and _plausible_year(struct.unpack("<Q", k[4:12])[0]) and _is_ascii(k[:4].rstrip(b"\x00"))
           for k, _ in keys) and all(len(v) % 24 == 0 or len(v) == 16 for _, v in keys):
        return "davies"
    if all(len(k) > 4 and _plausible_year(struct.unpack(">I", k[:4])[0]) for k, _ in keys):
        return "pivoted"
    if all(len(v) % 24 == 0 and v for _, v in keys) and all(_is_text(k) for k, _ in keys):
        return "ngram"
    return "raw"


# ---------------------------------------------------------------- reading
def rows(path: str, mode: str = "head", key: str = "", n: int = 20,
         layout: str = "auto", raw: bool = False) -> Tuple[str, List[tuple]]:
    """Up to ``n`` decoded records: the first ones (``head``), from ``key``
    onwards (``key``) or those starting with ``key`` (``prefix``). Returns
    the layout used and the rows as ``(key, value)`` strings, or with the
    raw key and value bytes appended when ``raw`` (for a year series)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}")
    p = Path(path).expanduser()
    if p.is_file():
        text_rows = _text_rows(p, mode, key, n)
        return "text", ([(k, v, k.encode(), v.encode()) for k, v in text_rows] if raw else text_rows)
    with _open(p) as db:
        if layout == "auto":
            layout = sniff_layout(db)
        start = b"" if mode == "head" else parse_key(key, layout)
        out: List[Row] = []
        for k, v in _scan(db, start):
            k = bytes(k)
            if mode == "prefix" and not k.startswith(start):
                break
            if layout != "raw" and _is_bookkeeping(k):
                continue
            v = bytes(v)
            shown = (format_key(k, layout), format_value(v, layout))
            out.append(shown + (k, v) if raw else shown)
            if len(out) >= n:
                break
    return layout, out


# ---------------------------------------------------------------- keys and values
def parse_key(text: str, layout: str) -> bytes:
    """The bytes a key typed as text stands for, in the given layout:
    ``[2011] nuclear`` or ``[2011]`` for pivoted, ``[fic][2000] quick`` for
    Davies, the n-gram itself otherwise. A pivoted key without a year
    prefix is taken as a prefix of every year's keys, which is not
    useful, so it is an error."""
    text = text.strip()
    if layout == "pivoted":
        if not text.startswith("["):
            raise ValueError("a pivoted database's keys start with the year: type [2011] nuclear, or [2011]")
        year, _, rest = text[1:].partition("]")
        try:
            return struct.pack(">I", int(year)) + rest.strip().encode("utf-8")
        except ValueError:
            raise ValueError(f"not a year: [{year}]") from None
    if layout == "davies":
        if not text.startswith("["):
            raise ValueError("a Davies database's keys start with the genre and year: type [fic][2000] text")
        genre, _, rest = text[1:].partition("]")
        year, _, rest = rest.lstrip("[").partition("]")
        try:
            return genre.encode("utf-8")[:4].ljust(4, b"\x00") + struct.pack("<Q", int(year)) + rest.strip().encode("utf-8")
        except ValueError:
            raise ValueError(f"not a year: [{year}]") from None
    if layout == "raw" and text.startswith("0x"):
        return bytes.fromhex(text[2:])
    return text.encode("utf-8")


def format_key(key: bytes, layout: str) -> str:
    if layout == "pivoted" and len(key) >= 4:
        return f"[{struct.unpack('>I', key[:4])[0]}] " + _text_or_hex(key[4:])
    if layout == "davies" and len(key) >= 12:
        genre = key[:4].rstrip(b"\x00").decode("utf-8", "replace")
        return f"[{genre}][{struct.unpack('<Q', key[4:12])[0]}] " + _text_or_hex(key[12:])
    if layout == "ngram":
        return _text_or_hex(key)
    return _text_or_hex(key) if _is_text(key) else "0x" + key.hex()


def format_value(value: bytes, layout: str) -> str:
    """Values of the three layouts are packed little-endian integers: a
    (year, count, volumes) triple per year for n-gram keys, one triple (or
    an older 16-byte count/volumes pair) for pivoted and Davies keys."""
    if layout in ("pivoted", "davies"):
        if len(value) == 24:
            _year, count, volumes = struct.unpack("<QQQ", value)
            return f"{count:,} in {volumes:,} volumes"
        if len(value) == 16:
            count, volumes = struct.unpack("<QQ", value)
            return f"{count:,} in {volumes:,} volumes"
    if layout in ("ngram", "pivoted", "davies") and value and len(value) % 24 == 0:
        triples = [struct.unpack("<QQQ", value[i:i + 24]) for i in range(0, len(value), 24)]
        count = sum(t[1] for t in triples)
        volumes = sum(t[2] for t in triples)
        years = [t[0] for t in triples]
        span = f"{years[0]}" if len(years) == 1 else f"{years[0]}-{years[-1]}, {len(years)} years"
        return f"{count:,} in {volumes:,} volumes ({span})"
    return _text_or_hex(value) if _is_text(value) else f"0x{value.hex()[:48]}{'...' if len(value) > 24 else ''}"


def year_series(value: bytes) -> List[Tuple[int, int, int]]:
    """The (year, count, volumes) triples of an n-gram-keyed value."""
    return [struct.unpack("<QQQ", value[i:i + 24]) for i in range(0, len(value) - len(value) % 24, 24)]


# ---------------------------------------------------------------- helpers
def _open(p: Path):
    from ngramprep.common_db.api import open_db
    return open_db(str(p), mode="r", profile="scan:packed24")


def _scan(db, start: bytes):
    it = db.iterator(fill_cache=False) if _accepts_fill_cache(db) else db.iterator()
    it.seek(start)
    while it.valid():
        yield it.key(), it.value()
        it.next()


def _accepts_fill_cache(db) -> bool:
    import inspect as _inspect                   # the stdlib one
    try:
        return "fill_cache" in _inspect.signature(db.iterator).parameters
    except (TypeError, ValueError):
        return True


def _bookkeeping_keys(db):
    for prefix in BOOKKEEPING_PREFIXES:
        for k, _ in _scan(db, prefix):
            if not bytes(k).startswith(prefix):
                break
            yield bytes(k)


def _is_bookkeeping(key: bytes) -> bool:
    return key.startswith(BOOKKEEPING_PREFIXES)


def _int_property(db, name: str) -> int:
    try:
        v = db.get_property(name)
        return int(v) if v is not None else 0
    except Exception:
        return 0


def _plausible_year(y: int) -> bool:
    return 1000 <= y <= 2200


def _is_ascii(b: bytes) -> bool:
    return all(32 <= c < 127 for c in b)


def _is_text(b: bytes) -> bool:
    try:
        s = b.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return all(c.isprintable() or c == " " for c in s)


def _text_or_hex(b: bytes) -> str:
    return b.decode("utf-8") if _is_text(b) else "0x" + b.hex()


def _text_rows(p: Path, mode: str, key: str, n: int) -> List[Row]:
    """A vocabulary file: rank and token. ``key`` finds a token (then the
    ones after it in rank order), ``prefix`` the tokens starting with it."""
    out: List[Row] = []
    key = key.strip()
    found = mode != "key"
    with open(p, "r", encoding="utf-8", errors="replace") as fh:
        for rank, line in enumerate(fh, 1):
            token = line.split()[0] if line.split() else ""
            if mode == "key" and not found:
                if token == key:
                    found = True
                else:
                    continue
            if mode == "prefix" and not token.startswith(key):
                continue
            out.append((f"{rank:,}", line.rstrip("\n")))
            if len(out) >= n:
                break
    return out


def candidates(corpus_dir: Optional[str]) -> List[Tuple[str, str]]:
    """Databases and vocabulary files in a corpus directory, as (label, path)."""
    if not corpus_dir:
        return []
    d = Path(corpus_dir).expanduser()
    if not d.is_dir():
        return []
    out = []
    for entry in sorted(d.iterdir()):
        if entry.is_dir() and entry.suffix == ".db" and (entry / "CURRENT").exists():
            out.append((entry.name, str(entry)))
            wl = entry / "whitelist.txt"
            if wl.is_file():
                out.append((f"{entry.name}/whitelist.txt", str(wl)))
    return out


def format_summary(info: Dict[str, Any]) -> str:
    if info.get("error"):
        return info["error"]
    if info["kind"] == "text":
        return f"vocabulary file: {info['lines']:,} lines, {_fmt_bytes(info['bytes'])}"
    state = "compacted" if info.get("compacted") else "not compacted (counts are estimates)"
    return (f"{info['records']:,} records, {_fmt_bytes(info['bytes'])}, {state}; "
            f"keys look {info['layout']}")


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"
