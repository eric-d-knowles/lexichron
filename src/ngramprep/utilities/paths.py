"""A directory written in a settings file, as the running process sees it.

A corpus directory may be written with environment variables and ``~``
(``/scratch/$USER/NLP_corpora/Google_Books``), so that one settings file
serves everyone on a cluster. They are expanded where the path is used,
inside the job that runs the stage. A variable that is not set is an
error: left as it is, it would become a directory literally named
``$USER``.
"""
from __future__ import annotations

import os
import re

__all__ = ["expand_path", "UnsetVariable"]

_VARIABLE = re.compile(r"\$(\{[A-Za-z_][A-Za-z0-9_]*\}|[A-Za-z_][A-Za-z0-9_]*)")


class UnsetVariable(ValueError):
    """A path names an environment variable that is not set."""


def expand_path(value) -> str:
    """``value`` with ``$VAR``, ``${VAR}`` and a leading ``~`` expanded.
    Raises :class:`UnsetVariable` if a variable it names is not set."""
    text = str(value)
    if "$" not in text and not text.startswith("~"):
        return text
    unset = sorted({m.group(1).strip("{}") for m in _VARIABLE.finditer(text)
                    if m.group(1).strip("{}") not in os.environ})
    if unset:
        raise UnsetVariable(f"{text}: the environment variable{'s' if len(unset) > 1 else ''} "
                            f"{', '.join('$' + v for v in unset)} {'are' if len(unset) > 1 else 'is'} not set")
    return os.path.expanduser(os.path.expandvars(text))
