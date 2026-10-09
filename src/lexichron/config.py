"""Project configuration files for the lexichron command line.

A project file is YAML with one ``corpus`` section shared by every stage and
one section per stage. Stage sections map directly onto the keyword arguments
of the stage's Python entry point, so the file documents the call that will be
made and nothing in it is interpreted twice.

Example::

    corpus:
      release: "20200217"
      language: eng-us
      db_path_stub: /scratch/me/NLP_corpora/Google_Books/

    acquire:
      ngram_size: 1
      ngram_type: tagged
      workers: 45
      overwrite_db: false

Settings can be overridden from the command line with ``--set`` using dotted
keys, e.g. ``--set acquire.ngram_size=2``.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Callable, Dict, Mapping

import yaml

__all__ = ["ConfigError", "load_project", "apply_overrides", "build_call"]


class ConfigError(ValueError):
    """A problem with a project file or a command-line override."""


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_project(path: str | Path) -> Dict[str, Any]:
    """Read a project YAML file into a nested dict."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"project file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping of sections")
    for name, section in data.items():
        if section is not None and not isinstance(section, dict):
            raise ConfigError(f"{path}: section '{name}' must be a mapping")
    return data


def apply_overrides(config: Dict[str, Any], overrides: list[str]) -> Dict[str, Any]:
    """Apply ``section.key=value`` overrides. Values are parsed as YAML, so
    ``workers=20`` is an int, ``overwrite_db=false`` a bool, and
    ``file_range=[0, 5]`` a list."""
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"override must look like section.key=value: {item}")
        dotted, _, raw = item.partition("=")
        parts = dotted.strip().split(".")
        if len(parts) != 2 or not all(parts):
            raise ConfigError(f"override key must be section.key: {dotted}")
        section, key = parts
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise ConfigError(f"cannot parse value for {dotted}: {raw} ({exc})") from exc
        config.setdefault(section, {})
        if config[section] is None:
            config[section] = {}
        config[section][key] = value
    return config


# ---------------------------------------------------------------------------
# Turning sections into function calls
# ---------------------------------------------------------------------------

# How ``corpus`` keys are named in the Python API.
CORPUS_KEY_MAP = {
    "release": "repo_release_id",
    "language": "repo_corpus_id",
    "db_path_stub": "db_path_stub",
    "archive_path_stub": "archive_path_stub",
}


def _coerce(name: str, value: Any, annotation: Any) -> Any:
    """Convert YAML-native values to what the API expects (lists to tuples or
    sets where the annotation asks for them)."""
    text = str(annotation)
    if isinstance(value, list):
        if "Tuple" in text or "tuple" in text:
            return tuple(value)
        if "Set" in text or "set" in text:
            return set(value)
    return value


def build_call(
    func: Callable[..., Any],
    config: Mapping[str, Any],
    stage: str,
) -> Dict[str, Any]:
    """Assemble keyword arguments for ``func`` from the ``corpus`` section and
    the ``stage`` section of ``config``.

    Every key is checked against the function's signature, so a misspelled
    setting fails here with a clear message rather than being ignored.
    """
    sig = inspect.signature(func)
    params = sig.parameters
    kwargs: Dict[str, Any] = {}

    corpus = config.get("corpus") or {}
    for key, value in corpus.items():
        if key not in CORPUS_KEY_MAP:
            raise ConfigError(
                f"unknown key in 'corpus' section: {key} "
                f"(expected one of {', '.join(CORPUS_KEY_MAP)})"
            )
        api_name = CORPUS_KEY_MAP[key]
        if api_name in params:
            kwargs[api_name] = _coerce(api_name, value, params[api_name].annotation)

    section = config.get(stage) or {}
    for key, value in section.items():
        if key not in params:
            close = [p for p in params if key.lower() in p.lower() or p.lower() in key.lower()]
            hint = f" (did you mean {', '.join(close)}?)" if close else ""
            raise ConfigError(f"unknown key in '{stage}' section: {key}{hint}")
        kwargs[key] = _coerce(key, value, params[key].annotation)

    missing = [
        name for name, p in params.items()
        if p.default is inspect.Parameter.empty and name not in kwargs
    ]
    if missing:
        where = {v: k for k, v in CORPUS_KEY_MAP.items()}
        pretty = [f"corpus.{where[m]}" if m in where else f"{stage}.{m}" for m in missing]
        raise ConfigError(f"missing required setting(s): {', '.join(pretty)}")

    return kwargs
