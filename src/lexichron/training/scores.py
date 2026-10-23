"""Where a tuning search keeps its scores.

Each scored cell-year is one small JSON file, ``scores/y1990_<cell>.json``,
and each scored year pair another, ``pairs/<cell>_y1968_y1969.json``. A
file is written whole and renamed into place, so one exists if and only if
its score does, and two processes scoring different things never touch the
same file: the tasks of a job array (``lexichron.arrays``) each write the
scores of their own years, and a year pair is written by whichever task
finds both years' models on disk. The tables people read (``results.tsv``,
``pairs.tsv``, ``summary.tsv``, ``summary.json``) are exports, rebuilt from
the files by whoever has something to add; readers derive from the files.

A search made before this layout has only the tables: ``import_tables``
turns them into files once, on the search's next run or read.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["ScoreStore", "write_tsv", "write_json", "read_tsv_rows", "cell_text"]

RESULTS_FILE = "results.tsv"      # one row per cell and year
PAIRS_FILE = "pairs.tsv"          # one row per cell and year pair
SUMMARY_FILE = "summary.tsv"      # one row per cell
SUMMARY_JSON = "summary.json"

RowKey = Tuple[str, int]
PairKey = Tuple[str, int, int]


class ScoreStore:
    def __init__(self, search_dir: str | os.PathLike) -> None:
        self.dir = Path(search_dir)
        self.scores_dir = self.dir / "scores"
        self.pairs_dir = self.dir / "pairs"

    # -- files -----------------------------------------------------------------
    @staticmethod
    def row_name(cell: str, year: int) -> str:
        return f"y{int(year)}_{cell}.json"

    @staticmethod
    def pair_name(cell: str, a: int, b: int) -> str:
        return f"{cell}_y{int(a)}_y{int(b)}.json"

    def has_files(self) -> bool:
        return self.scores_dir.is_dir()

    def import_tables(self) -> int:
        """A search written before the per-score files: its ``results.tsv``
        and ``pairs.tsv`` become files, once. Returns how many were written;
        0 when the files already exist or there is nothing to import."""
        if self.has_files():
            return 0
        rows = read_tsv_rows(self.dir / RESULTS_FILE)
        pairs = read_tsv_rows(self.dir / PAIRS_FILE)
        n = 0
        for r in rows:
            if "cell" in r and "year" in r:
                self.put_row(r)
                n += 1
        for p in pairs:
            if "cell" in p and "year_a" in p and "year_b" in p:
                self.put_pair(p)
                n += 1
        if n == 0 and rows == [] and pairs == []:
            return 0
        self.scores_dir.mkdir(parents=True, exist_ok=True)       # even if empty: the import is done
        return n

    def put_row(self, row: Dict[str, Any]) -> Path:
        path = self.scores_dir / self.row_name(row["cell"], int(row["year"]))
        _write_json(path, row)
        return path

    def put_pair(self, pair: Dict[str, Any]) -> Path:
        path = self.pairs_dir / self.pair_name(pair["cell"], int(pair["year_a"]), int(pair["year_b"]))
        _write_json(path, pair)
        return path

    def load_rows(self, years: Optional[Sequence[int]] = None) -> Dict[RowKey, Dict[str, Any]]:
        """Every scored cell-year (of ``years`` when given), from the files;
        from the table for a search that has not been imported."""
        wanted = {int(y) for y in years} if years is not None else None
        out: Dict[RowKey, Dict[str, Any]] = {}
        if self.has_files():
            for r in _read_json_dir(self.scores_dir):
                if "cell" in r and "year" in r:
                    out[(r["cell"], int(r["year"]))] = r
        else:
            for r in read_tsv_rows(self.dir / RESULTS_FILE):
                if "cell" in r and "year" in r:
                    out[(r["cell"], int(r["year"]))] = r
        if wanted is not None:
            out = {k: v for k, v in out.items() if k[1] in wanted}
        return out

    def load_pairs(self, years: Optional[Sequence[int]] = None) -> Dict[PairKey, Dict[str, Any]]:
        wanted = {int(y) for y in years} if years is not None else None
        out: Dict[PairKey, Dict[str, Any]] = {}
        if self.has_files():
            source = _read_json_dir(self.pairs_dir)
        else:
            source = read_tsv_rows(self.dir / PAIRS_FILE)
        for p in source:
            if "cell" in p and "year_a" in p and "year_b" in p:
                key = (p["cell"], int(p["year_a"]), int(p["year_b"]))
                if wanted is None or (key[1] in wanted and key[2] in wanted):
                    out[key] = p
        return out

    # -- exports ---------------------------------------------------------------
    def export(self, rows: Dict[RowKey, Dict[str, Any]], pairs: Dict[PairKey, Dict[str, Any]],
               summary: Dict[str, Any], tables: bool = True) -> None:
        """Write the tables from what the caller derived (usually from a
        fresh ``load_rows`` / ``load_pairs``, so a sibling task's additions
        are in them); with ``tables`` false only the summary."""
        if tables:
            write_tsv(self.dir / RESULTS_FILE, [rows[k] for k in sorted(rows, key=lambda k: (k[1], k[0]))])
            write_tsv(self.dir / PAIRS_FILE, [pairs[k] for k in sorted(pairs, key=lambda k: (k[1], k[2], k[0]))])
        write_tsv(self.dir / SUMMARY_FILE, summary.get("cells") or [])
        _write_json(self.dir / SUMMARY_JSON, summary)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    """Written whole under a temporary name and renamed into place (safe
    for several processes writing the same file)."""
    _write_json(path, data)


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.part")
    tmp.write_text(json.dumps(data, indent=1, default=_json_num))
    os.replace(tmp, path)


def _read_json_dir(d: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*.json")):
        try:
            data = json.loads(p.read_text())
        except (OSError, ValueError):
            continue                              # being written, or damaged: not a score
        if isinstance(data, dict):
            out.append({k: (float("nan") if v is None else v) for k, v in data.items()})
    return out


def _json_num(v):
    if isinstance(v, float) and v != v:
        return None
    raise TypeError(str(type(v)))


# -- tab-separated tables -----------------------------------------------------
def write_tsv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    cols: List[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    lines = ["\t".join(cols)]
    for r in rows:
        lines.append("\t".join(cell_text(r.get(c)) for c in cols))
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.part")
    tmp.write_text("\n".join(lines) + "\n")
    os.replace(tmp, path)


def cell_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return "" if v != v else f"{v:.4f}"
    return str(v)


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
