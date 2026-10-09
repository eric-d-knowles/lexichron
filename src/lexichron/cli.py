"""``lexichron`` command line.

    lexichron acquire project.yaml [--set acquire.ngram_size=2] [--dry-run]

Each subcommand reads the project file, merges any ``--set`` overrides, builds
the keyword arguments for the stage's Python entry point, and calls it. With
``--dry-run`` it prints the call instead.
"""
from __future__ import annotations

import argparse
import logging
import sys
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

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    logging.getLogger(__name__).info("lexichron %s: running stage '%s'", __version__, stage)
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

    for stage, (_, help_text) in STAGES.items():
        p = sub.add_parser(stage, help=help_text, description=help_text)
        p.add_argument("project", help="project YAML file")
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
    try:
        return _run_stage(args.stage, args)
    except ConfigError as exc:
        parser.exit(2, f"lexichron {args.stage}: {exc}\n")
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
