"""``lexichron`` command line.

    lexichron acquire lexichron.yaml [--set acquire.ngram_size=2] [--dry-run]

Each subcommand reads the project file, merges any ``--set`` overrides, builds
the keyword arguments for the stage's Python entry point, and calls it. With
``--dry-run`` it prints the call instead.
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


def _run_stage(stage: str, args: argparse.Namespace) -> int:
    dotted, _ = STAGES[stage]
    func = _resolve(dotted)

    config = load_project(args.project)
    apply_overrides(config, args.set or [])
    kwargs = build_call(func, config, stage)

    if args.dry_run:
        print(_format_call(func, kwargs))
        return 0

    # Progress document for this run, next to the project file, so that
    # `lexichron ui` and other tools can watch it.
    import inspect
    from datetime import datetime
    if "progress_path" in inspect.signature(func).parameters and "progress_path" not in kwargs:
        run_dir = (Path(args.project).resolve().parent / ".lexichron" / "runs"
                   / f"{datetime.now():%Y%m%d_%H%M%S}_{stage}")
        kwargs["progress_path"] = str(run_dir / "progress.json")
        print(f"Progress: {kwargs['progress_path']}", flush=True)

    # Logging is left to the stage: each pipeline writes a log file next to
    # its output and keeps the console to its banner, progress and warnings.
    print(f"lexichron {__version__}: running stage '{stage}'", flush=True)
    func(**kwargs)
    return 0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

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
