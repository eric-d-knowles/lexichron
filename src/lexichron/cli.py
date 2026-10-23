"""``lexichron`` command line.

    lexichron acquire lexichron.yaml [--set acquire.ngram_size=2] [--dry-run] [--yes]

Each subcommand reads the project file, merges any ``--set`` overrides, builds
the keyword arguments for the stage's Python entry point, and calls it. With
``--dry-run`` it prints the call instead. When the settings make the stage
delete earlier output (``acquire.overwrite_db``, ``filter.mode = restart``,
``tune.mode = restart``) it says what and how much and asks first; ``--yes``
answers for a batch job, which otherwise stops there.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Dict

from lexichron import __version__
from lexichron.config import ConfigError, apply_overrides, build_call, load_project

__all__ = ["main"]


# ---------------------------------------------------------------------------
# Stage registry: name -> (import path, description)
# Imports are deferred so that `lexichron --help` stays fast and so that a
# stage's heavy dependencies load only when that stage runs.
# ---------------------------------------------------------------------------
STAGES: Dict[str, tuple[str, str]] = {
    "acquire": (
        "ngramprep.ngram_acquire:download_and_ingest_to_rocksdb",
        "Download Google Books n-gram files and ingest them into RocksDB",
    ),
    "filter": (
        "lexichron.stages.filter:filter_ngrams",
        "Filter a raw n-gram database (lowercase, lemmatize, stopwords, vocabulary)",
    ),
    "tune": (
        "lexichron.stages.tune:tune_models",
        "Hyperparameter search: train a grid of word2vec models on sampled years and score their stability and quality",
    ),
}


def _resolve(dotted: str) -> Callable[..., Any]:
    module_name, _, attr = dotted.partition(":")
    module = __import__(module_name, fromlist=[attr])
    return getattr(module, attr)


def _format_call(func: Callable[..., Any], kwargs: Dict[str, Any]) -> str:
    lines = [f"{func.__module__}.{func.__name__}("]
    for key, value in kwargs.items():
        lines.append(f"    {key}={value!r},")
    lines.append(")")
    return "\n".join(lines)


def _confirm_overwrite(stage: str, kwargs: Dict[str, Any], *, assume_yes: bool = False) -> bool:
    """Before a stage deletes earlier output (see lexichron.safeguards), say
    what and how much, and ask. In a terminal the answer is typed; a batch
    job has nobody to ask, so it needs ``--yes`` and otherwise stops here,
    before any work is lost."""
    from lexichron.safeguards import overwrite_risk
    risk = overwrite_risk(stage, kwargs)
    if risk is None:
        return True
    print(risk.describe(), file=sys.stderr, flush=True)
    if assume_yes:
        print("Going ahead (--yes).", file=sys.stderr, flush=True)
        return True
    if not sys.stdin.isatty():
        print(f"lexichron {stage}: not deleting it without confirmation; run again with --yes "
              f"to go ahead, or {risk.keep_hint} to keep it.", file=sys.stderr, flush=True)
        return False
    try:
        answer = input("Delete it and start over? [y/N] ")
    except EOFError:
        answer = ""
    if answer.strip().lower() in ("y", "yes"):
        return True
    print(f"lexichron {stage}: nothing deleted, nothing run.", file=sys.stderr, flush=True)
    return False


def _run_stage(stage: str, args: argparse.Namespace) -> int:
    dotted, _ = STAGES[stage]
    func = _resolve(dotted)

    config = load_project(args.project)
    apply_overrides(config, args.set or [])
    kwargs = build_call(func, config, stage)

    if args.dry_run:
        print(_format_call(func, kwargs))
        return 0

    if not _confirm_overwrite(stage, kwargs, assume_yes=getattr(args, "yes", False)):
        return 2

    # Progress document for this run, next to the project file, so that
    # `lexichron ui` and other tools can watch it.
    import inspect
    from datetime import datetime
    if "progress_path" in inspect.signature(func).parameters and "progress_path" not in kwargs:
        # the tasks of a job array start in the same second: each gets its own
        # run directory, named by its task
        from lexichron.arrays import array_task
        try:
            task = array_task(kwargs.get("array_task"), kwargs.get("array_count"))
        except ValueError:
            task = None                                  # the stage reports it
        run_dir = (Path(args.project).resolve().parent / ".lexichron" / "runs"
                   / f"{datetime.now():%Y%m%d_%H%M%S}_{stage}{'_' + task.suffix if task else ''}")
        kwargs["progress_path"] = str(run_dir / "progress.json")
        print(f"Progress: {kwargs['progress_path']}", flush=True)

    # Logging is left to the stage: each pipeline writes a log file next to
    # its output and keeps the console to its banner, progress and warnings.
    print(f"lexichron {__version__}: running stage '{stage}'", flush=True)
    func(**kwargs)
    if kwargs.get("progress_path"):
        # The database carries its own record of how it was made (see
        # lexichron.provenance); `lexichron inspect` shows it.
        from lexichron.provenance import record_run
        try:
            written = record_run(stage, kwargs, kwargs["progress_path"])
        except Exception as exc:                        # never fail a finished run over this
            print(f"lexichron {stage}: could not write the provenance record: {exc}", file=sys.stderr)
        else:
            if written:
                print(f"Provenance: {written}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _inspect(args: argparse.Namespace) -> int:
    from lexichron.inspection import describe, format_summary, rows, year_series
    from lexichron.provenance import format_provenance
    info = describe(args.path)
    print(format_summary(info))
    if info.get("error"):
        return 1
    if args.info:
        prov = info.get("provenance")
        if not prov:
            print("(no provenance record: made before lexichron 0.6, or not by lexichron)")
            return 0
        for line in format_provenance(prov):
            print(line)
        return 0
    if info["kind"] == "search":                       # a tuning search: its per-cell table
        from lexichron.inspection import cell_detail, search_table
        headings, grid, stems = search_table(info["search"])
        if not grid:
            print("(no cells scored yet)")
            return 0
        widths = [max(len(headings[i]), *(len(r[i]) for r in grid)) for i in range(len(headings))]
        for row in [headings] + grid:
            print("  ".join(f"{c:<{w}}" for c, w in zip(row, widths)).rstrip())
        if args.key:                                   # a cell stem: its per-year rows
            for line in cell_detail(info["search"], args.key):
                print(line)
        return 0
    mode = "key" if args.key is not None else "prefix" if args.prefix is not None else "head"
    try:
        layout, found = rows(args.path, mode, args.key or args.prefix or "", args.rows, args.layout, raw=True)
    except ValueError as exc:
        print(f"lexichron inspect: {exc}", file=sys.stderr)
        return 2
    if not found:
        print("(no records)")
        return 0
    if layout == "table":
        grid = [[k] + v.split("\t") if v else k.split("\t") for k, v, _rk, _rv in found]
        widths = [max(len(row[i]) if i < len(row) else 0 for row in grid) for i in range(max(len(r) for r in grid))]
        for row in grid:
            print("  ".join(f"{c:<{w}}" for c, w in zip(row, widths)).rstrip())
        return 0
    width = min(max(len(k) for k, _, _, _ in found), 60)
    for k, v, _rk, rv in found:
        print(f"{k:<{width}}  {v}")
        if args.series and layout == "ngram":
            for year, count, volumes in year_series(rv):
                print(f"{'':<{width}}    {year}: {count:,} in {volumes:,} volumes")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lexichron",
        description="Run lexichron pipeline stages from a project file.",
    )
    parser.add_argument("--version", action="version", version=f"lexichron {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="STAGE")
    sub.required = True

    ui = sub.add_parser("ui", help="Open the terminal user interface",
                        description="Open the terminal user interface (requires the 'ui' extra).")
    ui.add_argument("ui_args", nargs=argparse.REMAINDER, help="arguments for the UI (project file, --stage)")
    ui.set_defaults(stage=None, command="ui")

    ins = sub.add_parser(
        "inspect", help="Show records of a database or vocabulary file",
        description="Show a few records of a database (opened read-only) or a vocabulary file: the "
                    "first ones, those from a key onwards, or those with a prefix. Keys are decoded "
                    "for the file's layout: [2011] nuclear for a pivoted database, the n-gram for "
                    "an n-gram-keyed one, [fic][2000] text for a Davies corpus.")
    ins.add_argument("path", help="database directory or vocabulary file")
    which = ins.add_mutually_exclusive_group()
    which.add_argument("--key", metavar="KEY", help="records from this key onwards (for a tuning search: a cell, for its per-year rows)")
    which.add_argument("--prefix", metavar="PREFIX", help="records whose key starts with this")
    ins.add_argument("--rows", type=int, default=20, metavar="N", help="how many records (default 20)")
    ins.add_argument("--layout", choices=("auto", "pivoted", "ngram", "davies", "raw"), default="auto",
                     help="key layout (default: sniffed from the first records)")
    ins.add_argument("--series", action="store_true",
                     help="for n-gram-keyed records, print the full year series under each row")
    ins.add_argument("--info", action="store_true",
                     help="print the database's provenance record (how it was made) and exit")
    ins.set_defaults(stage=None, command="inspect")

    for stage, (_, help_text) in STAGES.items():
        p = sub.add_parser(stage, help=help_text, description=help_text)
        p.add_argument("project", help="settings YAML file (e.g. lexichron.yaml)")
        p.add_argument(
            "--set", action="append", metavar="SECTION.KEY=VALUE",
            help="override a setting from the project file (repeatable)",
        )
        p.add_argument(
            "--dry-run", action="store_true",
            help="print the resolved call and exit without running it",
        )
        p.add_argument(
            "--yes", "-y", action="store_true",
            help="go ahead without asking when the settings delete earlier output "
                 "(acquire.overwrite_db, filter.mode = restart, tune.mode = restart); "
                 "a batch job cannot be asked and stops without this",
        )
        p.set_defaults(stage=stage)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "ui":
        try:
            from lexichron.ui.app import main as ui_main
        except ImportError as exc:  # textual missing
            parser.exit(1, f"lexichron ui: the terminal UI needs the 'ui' extra "
                           f"(pip install 'lexichron[ui]'): {exc}\n")
        return ui_main(args.ui_args)
    if args.command == "inspect":
        return _inspect(args)
    try:
        return _run_stage(args.stage, args)
    except ConfigError as exc:
        parser.exit(2, f"lexichron {args.stage}: {exc}\n")
    except KeyboardInterrupt:
        return 130
    except RuntimeError as exc:
        # A stage reporting failure (e.g. AcquisitionError): message, not a traceback
        print(f"lexichron {args.stage}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
