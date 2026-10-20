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
    if p.is_dir() and (p / "search.json").is_file():
        search = read_search(str(p))
        return {"kind": "search", "layout": "search", "search": search, "provenance": search.get("provenance")}
    if p.is_file():
        try:
            with open(p, "rb") as fh:
                lines = sum(1 for _ in fh)
        except OSError as exc:
            return {"kind": "text", "error": str(exc)}
        if p.suffix.lower() == ".tsv":
            cols = read_table(p, 0)[0]
            info = {"kind": "table", "lines": max(lines - 1, 0), "columns": cols, "bytes": p.stat().st_size,
                    "layout": "table"}
            search = read_search_summary(p.parent)
            if search:
                info["search"] = search
            return info
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
    if p.is_file() and p.suffix.lower() == ".tsv":
        cols, table = read_table(p, n, mode, key)
        out = [("\t".join(cols), "")] + [(r[0], "\t".join(r[1:])) for r in table]
        return "table", ([(k, v, k.encode(), v.encode()) for k, v in out] if raw else out)
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


def read_table(p: Path, n: int, mode: str = "head", key: str = "") -> Tuple[List[str], List[List[str]]]:
    """A TSV's header and up to ``n`` rows (``key``: rows from the first
    whose first column equals it; ``prefix``: rows whose first column starts
    with it). ``n`` 0 reads the header only."""
    cols: List[str] = []
    rows: List[List[str]] = []
    key = key.strip()
    found = mode != "key"
    with open(p, encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            parts = line.rstrip("\n").split("\t")
            if i == 0:
                cols = parts
                continue
            if n <= 0:
                break
            if mode == "key" and not found:
                if parts[0] == key:
                    found = True
                else:
                    continue
            if mode == "prefix" and not parts[0].startswith(key):
                continue
            rows.append(parts)
            if len(rows) >= n:
                break
    return cols, rows


def read_search_summary(search_dir: Path) -> Optional[Dict[str, Any]]:
    """The suggested cell of a tuning search, from its summary.json."""
    import json
    try:
        d = json.loads((Path(search_dir) / "summary.json").read_text())
    except (OSError, ValueError):
        return None
    return {"best_cell": d.get("best_cell"), "best_rule": d.get("best_rule"), "cells": len(d.get("cells") or [])}


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
    tuning = d / "tuning"
    if tuning.is_dir():
        for search in sorted(tuning.iterdir()):
            for name in ("summary.tsv", "results.tsv", "pairs.tsv"):
                if (search / name).is_file():
                    out.append((f"tuning/{search.name}/{name}", str(search / name)))
    return out


def format_summary(info: Dict[str, Any]) -> str:
    if info.get("error"):
        return info["error"]
    if info["kind"] == "search":
        return "\n".join(format_search(info["search"]))
    if info["kind"] == "table":
        text = f"table: {info['lines']:,} rows, {len(info['columns'])} columns"
        if info.get("search") and info["search"].get("best_cell"):
            text += f"; suggested cell {info['search']['best_cell']} ({info['search']['best_rule']})"
        return text
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


# ---------------------------------------------------------------- tuning searches
# the summary table's columns: (key in summary.json / summary.tsv, heading, format)
SEARCH_COLUMNS: Tuple[Tuple[str, str, str], ...] = (
    ("vector_size", "Dims", "int"), ("window", "Window", "int"), ("min_count", "Min count", "int"),
    ("epochs", "Epochs", "int"), ("approach", "Approach", "str"), ("weight_by", "Weighting", "str"),
    ("years", "Years", "int"), ("vocab_size", "Vocab", "int"), ("coverage", "Coverage", "num"),
    ("simlex_rho", "SimLex ρ", "num"), ("wordsim_rho", "WordSim ρ", "num"), ("analogy_accuracy", "Analogy", "num"),
    ("seed_overlap", "Seed", "num"), ("split_half_overlap", "Split-half", "num"),
    ("adjacent_overlap", "Adjacent", "num"), ("distant_overlap", "Distant", "num"),
    ("signal_to_noise", "Signal/noise", "num"),
)
STABILITY_NOTE = ("Seed, split-half and adjacent: mean overlap of the ten nearest neighbours of probe words "
                  "between two models (1 = identical neighbourhoods); distant: the same between the first and "
                  "last sampled years; signal/noise: change over the span relative to change over a year.")


def searches(corpus_dir: Optional[str]) -> List[Tuple[str, str]]:
    """Tuning searches of a corpus directory, newest first, as (name, path)."""
    if not corpus_dir:
        return []
    tuning = Path(corpus_dir).expanduser() / "tuning"
    if not tuning.is_dir():
        return []
    found = [s for s in tuning.iterdir() if s.is_dir() and (s / "search.json").is_file()]
    found.sort(key=lambda s: (s / "search.json").stat().st_mtime, reverse=True)
    return [(s.name, str(s)) for s in found]


def read_search(search_dir: str) -> Dict[str, Any]:
    """Everything a tuning search has written so far: its settings
    (``search.json``), the per-cell summary (``summary.json``, with the
    suggested cell), the per-year rows (``results.tsv``), the year pairs
    (``pairs.tsv``), how far it has got, and its provenance when finished."""
    import json
    d = Path(search_dir).expanduser()
    out: Dict[str, Any] = {"path": str(d), "name": d.name, "settings": {}, "cells": [], "best_cell": None,
                           "best_rule": "", "results": [], "pairs": [], "years_sampled": [], "years_scored": [],
                           "models_on_disk": 0, "provenance": None}
    if not d.is_dir():
        out["error"] = f"{d}: no such directory"
        return out
    try:
        out["settings"] = json.loads((d / "search.json").read_text())
    except (OSError, ValueError):
        out["error"] = f"{d}: not a tuning search (no search.json)"
        return out
    out["years_sampled"] = [int(y) for y in out["settings"].get("sample_years") or []]
    try:
        summary = json.loads((d / "summary.json").read_text())
        out["cells"] = list(summary.get("cells") or [])
        out["best_cell"], out["best_rule"] = summary.get("best_cell"), summary.get("best_rule") or ""
    except (OSError, ValueError):
        pass
    out["results"] = read_tsv_rows(d / "results.tsv")
    out["pairs"] = read_tsv_rows(d / "pairs.tsv")
    out["years_scored"] = sorted({int(r["year"]) for r in out["results"] if "year" in r})
    # The per-cell table is rebuilt from the rows, so it covers every scored
    # cell whatever version of the stage wrote summary.json (versions before
    # 0.8.2 summarized only the current grid's cells); the file is the
    # fallback when the rows cannot be summarized.
    if out["results"]:
        try:
            from lexichron.stages.tune import summarize_results
            from lexichron.training.grid import Cell
            rows = {(r["cell"], int(r["year"])): r for r in out["results"] if "cell" in r and "year" in r}
            cells = []
            for stem in sorted({c for c, _ in rows}):
                try:
                    cells.append(Cell.from_stem(stem))
                except Exception:
                    pass
            summary = summarize_results(rows, out["pairs"], cells)
            if len(summary.get("cells") or []) > len(out["cells"]):      # the file knows fewer cells
                out["cells"] = summary["cells"]
                out["best_cell"], out["best_rule"] = summary.get("best_cell"), summary.get("best_rule") or ""
        except Exception:
            pass
    models = d / "models"
    out["models_on_disk"] = sum(1 for _ in models.glob("*.npz")) if models.is_dir() else 0
    from lexichron.provenance import read_provenance
    out["provenance"] = read_provenance(d)
    return out


def read_tsv_rows(path: Path) -> List[Dict[str, Any]]:
    """Rows of a tab-separated table with a header, numbers restored (blank
    cells are NaN); [] when the file is missing or unreadable."""
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return []
    if not lines:
        return []
    cols = lines[0].split("\t")
    return [{c: _typed(v) for c, v in zip(cols, line.split("\t"))} for line in lines[1:] if line.strip()]


def _typed(text: str) -> Any:
    if text == "":
        return float("nan")
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            continue
    return text


PARAM_KEYS = ("vector_size", "window", "min_count", "epochs", "approach", "weight_by")
SCORE_KEYS = tuple(k for k, _, _ in SEARCH_COLUMNS if k not in PARAM_KEYS)
YEAR_KEYS = ("vocab_size", "coverage", "simlex_rho", "wordsim_rho", "analogy_accuracy", "seed_overlap",
             "split_half_overlap")


def search_overview(search: Dict[str, Any]) -> Dict[str, Any]:
    """The search at a glance: ``tiles`` [(label, value, note)], a ``grid``
    line, and the suggestion's ``rule``."""
    if search.get("error"):
        return {"tiles": [], "grid": search["error"], "rule": ""}
    s = search["settings"]
    grid = s.get("grid") or {}
    n_grid = 1
    for vals in grid.values():
        if isinstance(vals, (list, tuple)) and vals:
            n_grid *= len(vals)
    axes = ", ".join(f"{k.replace('_', ' ')} {'/'.join(str(v) for v in vals)}" for k, vals in grid.items()
                     if isinstance(vals, (list, tuple)) and len(vals) > 1)
    fixed = ", ".join(f"{k.replace('_', ' ')} {vals[0]}" for k, vals in grid.items()
                      if isinstance(vals, (list, tuple)) and len(vals) == 1)
    sampled, scored = search["years_sampled"], search["years_scored"]
    seeds, halves = int(s.get("seeds", 1) or 1), bool(s.get("split_half", False))
    per_cell = seeds + (2 if halves else 0)
    # finished only when every cell of the current grid has every sampled year
    try:
        from lexichron.training.grid import Grid
        grid_stems = {c.stem for c in Grid.from_settings(grid).cells()} if grid else set()
    except Exception:
        grid_stems = set()
    in_grid = [c for c in search["cells"] if not grid_stems or c.get("cell") in grid_stems]
    cells_done = sum(1 for c in in_grid if int(c.get("years", 0) or 0) >= len(sampled))
    done = bool(sampled) and len(scored) >= len(sampled) and cells_done >= n_grid
    partial = n_grid - cells_done
    years_note = ("finished" if done else
                  f"in progress; {partial} cell(s) not yet on all years" if scored and partial and
                  len(scored) >= len(sampled) else "in progress" if scored else "nothing scored yet")
    n_cells = len(search["cells"])
    tiles = [("Cells", f"{n_cells}", f"of {n_grid} in the grid" if n_cells <= n_grid
              else f"scored; {n_grid} in the current grid"),
             ("Years", f"{len(scored)}/{len(sampled)}" if sampled else "–", years_note),
             ("Models", f"{search['models_on_disk']:,}", f"on disk; {per_cell} per cell and year"),
             ("Suggested cell", search.get("best_cell") or "–", "")]
    years_text = (f"{sampled[0]}–{sampled[-1]}, {len(sampled)} sampled" if sampled else "?")
    grid_line = (f"Grid: {axes or 'one cell'}" + (f"  ·  fixed: {fixed}" if fixed else "")
                 + f"  ·  {seeds} seed(s)" + (" + split halves" if halves else "") + f"  ·  years {years_text}")
    if not done and scored:
        grid_line += "  ·  provisional: " + ", ".join(str(y) for y in scored) + " scored so far"
    return {"tiles": tiles, "grid": grid_line, "rule": search.get("best_rule") or ""}


def format_search(search: Dict[str, Any]) -> List[str]:
    """Header lines for the terminal."""
    if search.get("error"):
        return [search["error"]]
    o = search_overview(search)
    lines = [f"Search '{search['name']}': " + "; ".join(f"{label} {value} ({note})" if note else f"{label} {value}"
                                                       for label, value, note in o["tiles"][:3]),
             o["grid"]]
    if search.get("best_cell"):
        lines.append(f"Suggested cell: {search['best_cell']} — {o['rule']}")
    return lines


def search_table(search: Dict[str, Any]) -> Tuple[List[str], List[List[str]], List[str]]:
    """The per-cell summary as (headings, rows, cell stems): the parameters
    the grid varies, then the scores; the suggested cell marked ★."""
    grid = (search.get("settings") or {}).get("grid") or {}
    cells = search.get("cells") or []
    # a parameter column is shown when the grid varies it or the rows differ on it
    shown = [(k, h, kind) for k, h, kind in SEARCH_COLUMNS
             if k not in PARAM_KEYS or len(grid.get(k) or []) > 1 or len({c.get(k) for c in cells}) > 1]
    headings = ["", *(h for _, h, _ in shown)]
    rows, stems = [], []
    for c in cells:
        rows.append(["★" if c.get("cell") == search.get("best_cell") else ""]
                    + [_fmt_cell(c.get(key), kind) for key, _, kind in shown])
        stems.append(str(c.get("cell")))
    return headings, rows, stems


def best_in_columns(search: Dict[str, Any], headings: List[str]) -> Dict[str, Any]:
    """For each score heading, the best (highest) value among the cells when
    one cell alone reaches it, so a display can mark it."""
    cells = search.get("cells") or []
    out: Dict[str, Any] = {}
    for key, h, kind in SEARCH_COLUMNS:
        if h in headings and key in SCORE_KEYS and key not in ("years", "vocab_size"):
            vals = [c.get(key) for c in cells if isinstance(c.get(key), (int, float)) and c.get(key) == c.get(key)]
            if vals and vals.count(max(vals)) == 1:
                out[h] = _fmt_cell(max(vals), kind)
    return out


def cell_years(search: Dict[str, Any], stem: str) -> Tuple[List[str], List[List[str]]]:
    """One cell's per-year rows as (headings, rows)."""
    rows = sorted((r for r in search.get("results") or [] if r.get("cell") == stem), key=lambda r: r.get("year", 0))
    keys = [k for k in YEAR_KEYS if any(k in r for r in rows)]
    labels = {key: h for key, h, _ in SEARCH_COLUMNS}
    headings = ["Year"] + [labels.get(k, k) for k in keys]
    out = [[str(r.get("year"))] + [_fmt_cell(r.get(k), "int" if k == "vocab_size" else "num") for k in keys]
           for r in rows]
    return headings, out


def cell_pairs(search: Dict[str, Any], stem: str) -> str:
    """One cell's year pairs as a line."""
    pairs = [p for p in search.get("pairs") or [] if p.get("cell") == stem]
    if not pairs:
        return ""
    return "Year pairs (neighbour overlap / aligned cosine):  " + "   ".join(
        f"{p.get('kind')} {p.get('year_a')}–{p.get('year_b')} {_fmt_cell(p.get('overlap'), 'num')} / "
        f"{_fmt_cell(p.get('cosine'), 'num')}" for p in pairs)


def cell_detail(search: Dict[str, Any], stem: str) -> List[str]:
    """One cell's per-year rows and year pairs, as lines for the terminal."""
    headings, rows = cell_years(search, stem)
    pairs = cell_pairs(search, stem)
    if not rows and not pairs:
        return [f"{stem}: no rows yet"]
    lines = [f"{stem}: per year"]
    for r in rows:
        lines.append("  " + r[0] + ": " + "  ".join(f"{h} {v}" for h, v in zip(headings[1:], r[1:])))
    if pairs:
        lines.append("  " + pairs)
    return lines


def _fmt_cell(v: Any, kind: str) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "–"
    if kind == "int":
        try:
            return f"{int(v):,}"
        except (TypeError, ValueError):
            return str(v)
    if kind == "num":
        try:
            return f"{float(v):.3f}"
        except (TypeError, ValueError):
            return str(v)
    return str(v)
