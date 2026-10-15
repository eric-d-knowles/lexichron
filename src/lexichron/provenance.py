"""Provenance: a ``lexichron.json`` written into every database a stage
produces, saying what made it, from what, with which settings, and the
figures of the run. It travels with the database (copy or archive the
directory and the record comes along), unlike the progress records under
``.lexichron/runs/``, which are working files.

The filter stage's record includes a digest of the raw database's record,
so a pivoted database describes its whole chain back to Google's files.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PROVENANCE_FILENAME = "lexichron.json"
FORMAT = 1


def record_run(stage: str, settings: Dict[str, Any], progress_path: str) -> Optional[Path]:
    """Write the provenance of a finished run from its progress document and
    the resolved settings it ran with. Returns the file written, or None
    when the run did not finish or wrote no database."""
    from ngramprep.ngram_acquire.progress import read_progress
    doc = read_progress(progress_path) or {}
    if doc.get("state") != "done" or not doc.get("db_path"):
        return None
    db_dir = Path(doc["db_path"])
    if not db_dir.is_dir():
        return None
    record = build_record(stage, settings, dict(doc, _path=str(progress_path)))
    return write_provenance(db_dir, record)


def build_record(stage: str, settings: Dict[str, Any], doc: Dict[str, Any]) -> Dict[str, Any]:
    from lexichron import __version__
    try:
        from lexichron._build import BUILD
    except Exception:
        BUILD = None
    try:
        import importlib.metadata as md
        rocks = md.version("rocks-shim")
    except Exception:
        rocks = None
    counters = dict(doc.get("counters") or {})
    figures: Dict[str, Any] = {}
    if stage == "acquire":
        corpus = doc.get("corpus") or {}
        figures = {"files": int(doc.get("files_total", 0)),
                   "lines_read": int(corpus.get("lines", counters.get("lines", 0))),
                   "uncompressed_bytes": float(corpus.get("bytes", doc.get("uncompressed_bytes", 0))),
                   "inserted": int(corpus.get("entries", counters.get("inserted", 0)))}
    else:
        figures = {k: v for k, v in counters.items()}
        figures.setdefault("work_units", int(doc.get("files_total", 0)))
    database = dict(doc.get("database") or {})
    database["path"] = doc.get("db_path")
    inputs = _describe_inputs(stage, settings, doc)
    started, finished = doc.get("started"), doc.get("finished")
    elapsed = None
    try:
        if started and finished:
            elapsed = (datetime.fromisoformat(finished) - datetime.fromisoformat(started)).total_seconds()
    except ValueError:
        pass
    return {
        "format": FORMAT,
        "stage": stage,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "lexichron": __version__,
        "build": BUILD,
        "rocks_shim": rocks,
        "hostname": doc.get("hostname") or socket.gethostname(),
        "slurm_job_id": doc.get("slurm_job_id") or os.environ.get("SLURM_JOB_ID"),
        "settings": _json_safe({k: v for k, v in settings.items() if k != "progress_path"}),
        "inputs": inputs,
        "figures": _json_safe(figures),
        "database": _json_safe(database),
        "run": {"started": started, "finished": finished, "elapsed_s": elapsed,
                "progress": doc.get("_path")},
    }


def _describe_inputs(stage: str, settings: Dict[str, Any], doc: Dict[str, Any]) -> Dict[str, Any]:
    raw = dict(doc.get("inputs") or {})
    out: Dict[str, Any] = {}
    source = raw.get("source")
    if source:
        entry: Dict[str, Any] = {"path" if os.path.exists(str(source)) else "description": str(source)}
        upstream = read_provenance(source) if os.path.isdir(str(source)) else None
        if upstream:
            entry["provenance"] = summarize_upstream(upstream, digest(Path(source) / PROVENANCE_FILENAME))
        out["source"] = entry
    vocab = raw.get("vocabulary")
    if vocab:
        entry = {"path": str(vocab)}
        try:
            with open(vocab, "rb") as fh:
                data = fh.read()
            entry["tokens"] = sum(1 for line in data.splitlines() if line.strip())
            entry["sha256"] = hashlib.sha256(data).hexdigest()
        except OSError:
            pass
        out["vocabulary"] = entry
    elif stage == "filter" and "vocabulary" in raw:
        out["vocabulary"] = None                        # rule filters only
    if raw.get("always_include"):
        out["always_include"] = str(raw["always_include"])
    for key, value in raw.items():
        if key in ("source", "vocabulary", "always_include") or not value:
            continue
        if os.path.isfile(str(value)):                  # a dataset or other file: fingerprint it
            out[key] = {"path": str(value), "sha256": digest(Path(str(value)))}
        else:
            out[key] = str(value)
    return out


def summarize_upstream(upstream: Dict[str, Any], sha: Optional[str]) -> Dict[str, Any]:
    """What a downstream record keeps of its input's record."""
    return {"stage": upstream.get("stage"), "created": upstream.get("created"),
            "lexichron": upstream.get("lexichron"), "sha256": sha,
            "figures": upstream.get("figures"), "database": upstream.get("database")}


def write_provenance(db_dir: Path, record: Dict[str, Any]) -> Path:
    path = Path(db_dir) / PROVENANCE_FILENAME
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=False) + "\n")
    os.replace(tmp, path)
    return path


def read_provenance(db_dir: str | os.PathLike) -> Optional[Dict[str, Any]]:
    path = Path(db_dir) / PROVENANCE_FILENAME
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def digest(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def format_provenance(p: Dict[str, Any]) -> List[str]:
    """Lines for a panel or the terminal."""
    lines = []
    when = (p.get("run") or {}).get("finished") or p.get("created") or "?"
    elapsed = (p.get("run") or {}).get("elapsed_s")
    where = p.get("hostname") or "?"
    job = p.get("slurm_job_id")
    lines.append(f"Made by lexichron {p.get('lexichron')}" + (f" ({p['build']})" if p.get("build") else "")
                 + f", stage {p.get('stage')}, finished {when[:19].replace('T', ' ')}"
                 + (f" after {_fmt_duration(elapsed)}" if elapsed else "")
                 + f" on {where}" + (f", Slurm job {job}" if job else ""))
    inputs = p.get("inputs") or {}
    src = inputs.get("source") or {}
    if src:
        up = src.get("provenance")
        lines.append("Source: " + (src.get("path") or src.get("description") or "?")
                     + (f"  [made by stage {up['stage']} on {str(up.get('created', ''))[:10]}, record sha256 {up['sha256'][:12]}…]"
                        if up and up.get("sha256") else ""))
    if "vocabulary" in inputs:
        v = inputs["vocabulary"]
        lines.append("Vocabulary: " + (f"{v['path']} ({v.get('tokens', '?'):,} tokens, sha256 {v['sha256'][:12]}…)"
                                       if v and v.get("sha256") else (v["path"] if v else "none (rule filters only)")))
    if inputs.get("always_include"):
        lines.append(f"Always kept: {inputs['always_include']}")
    for key, v in inputs.items():
        if key.endswith("_dataset") and isinstance(v, dict):
            lines.append(f"{key.replace('_', ' ').capitalize()}: {v['path']}"
                         + (f" (sha256 {v['sha256'][:12]}…)" if v.get("sha256") else ""))
    fig = p.get("figures") or {}
    if p.get("stage") == "acquire":
        lines.append(f"Figures: {fig.get('files', 0):,} files, {fig.get('lines_read', 0):,} lines read, "
                     f"{_fmt_bytes(fig.get('uncompressed_bytes', 0))} parsed, {fig.get('inserted', 0):,} inserted")
    else:
        parts = [f"{fig.get('scanned', 0):,} scanned", f"{fig.get('kept', 0):,} kept"]
        if fig.get("produced", 0) > fig.get("kept", 0):
            parts.append(f"{fig['produced']:,} year records")
        parts.append(f"{fig.get('inserted', 0):,} inserted")
        lines.append("Figures: " + ", ".join(parts))
        rules = [(label, fig.get(key)) for label, key in (("not alphabetic", "tokens_not_alpha"),
                                                          ("too short", "tokens_too_short"),
                                                          ("stopword", "tokens_stopword"),
                                                          ("not in vocabulary", "tokens_not_in_vocab"),
                                                          ("n-grams with too few real tokens", "ngrams_dropped_context"))
                 if fig.get(key)]
        if fig.get("tokens"):
            lines.append(f"Dropped: {fig['tokens']:,} tokens seen; " + ", ".join(f"{n:,} {label}" for label, n in rules))
    db = p.get("database") or {}
    if db.get("records") is not None:
        lines.append(f"Database: {db['records']:,} records, {_fmt_bytes(db.get('bytes', 0))}"
                     + (", compacted" if db.get("compacted") else ", not compacted"))
    settings = p.get("settings") or {}
    if settings:
        shown = ", ".join(f"{k}={_short(v)}" for k, v in settings.items()
                          if k not in ("db_path_stub", "repo_release_id", "repo_corpus_id", "spool_dir", "archive_path_stub"))
        lines.append("Settings: " + shown)
    return lines


def _short(v: Any) -> str:
    s = json.dumps(v) if not isinstance(v, str) else v
    return s if len(s) <= 60 else s[:57] + "..."


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _fmt_bytes(n: float) -> str:
    n = float(n or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _fmt_duration(s: float) -> str:
    s = int(s)
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    return f"{h}h{m:02d}m" if h else (f"{m}m{sec:02d}s" if m else f"{sec}s")
