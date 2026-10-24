"""``lexichron ui``: a terminal interface for configuring, running and
submitting pipeline stages from a settings file.

Terminology: the *settings file* (default ``lexichron.yaml`` in the corpus's
own directory, ``<stub>/<release>/<language>/<n>gram_files/``) holds the corpus, the stage settings and the Slurm resources; it is
what ``lexichron <stage> file.yaml`` runs from. It is not a "project
environment" (what ``new-project`` creates), which the UI does not need.

Layout: a tab per concern.
  Settings — form for the settings file (generated from the stage's signature;
             rarely-used settings under a collapsed "Advanced" heading, and the
             resolved call exactly as --dry-run prints it under another)
  Run      — run the stage here (login node test, or inside an allocation)
  Submit   — Slurm resources; writes the batch script and submits it (sbatch
             on this machine, or through the host bridge from inside the container)
  Progress — the latest runs (from each run's progress.json) with a progress
             bar, rate, time left and the tail of the run's log; plus squeue
             for your Slurm jobs
"""
from __future__ import annotations

import getpass
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml
from rich.text import Text
from textual import on, work
from textual.message import Message
from textual.app import App, ComposeResult
from textual.containers import Grid, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button, Collapsible, DataTable, Footer, Header, Input, Label, ProgressBar,
    RichLog, Select, Static, Switch, TabbedContent, TabPane,
)

from lexichron import __version__
from lexichron.bridge import BridgeUnavailable, SlurmRunner, slurm_backend
from lexichron.cli import STAGES, _format_call, _resolve
from lexichron.config import ConfigError, build_call, load_project
from lexichron.corpora import CorpusInfo, corpus_info, scan_corpora, stage_prerequisites
from lexichron.inspection import (ENSEMBLE_LEGEND, FAMILY_LABELS, LAYOUTS, PARAM_KEYS, SEARCH_COLUMNS, STABILITY_NOTE,
                                  best_in_columns, ensemble_overview, ensemble_profile, ensemble_table, ensembles,
                                  read_ensemble,
                                  suggested_for, candidates, cell_pairs, cell_years,
                                  describe, format_summary, read_search, rows as inspect_rows, search_overview,
                                  search_table, searches, year_series)
from lexichron.provenance import format_provenance
from lexichron.resources import MemoryEstimate, memory_estimate
from lexichron.safeguards import OverwriteRisk, overwrite_risk, overwrite_target
from lexichron.training.grid import AXIS_BY_NAME, Grid as HyperGrid
from lexichron.ui.progress_view import fmt_count, fmt_duration, meters, summarize, tail_lines
from lexichron.ui.schema import CHOICES, Field, SLURM_FIELDS, Section, stage_sections
from lexichron.slurm_settings import (STAGE_KEYS, USER_KEYS, describe_sources, read_user_defaults,
                                      resolve as resolve_slurm, seed_user_defaults, stage_override,
                                      state_dir as _state_dir, write_user_defaults)
from lexichron.ui.slurm import ARRAY_STAGES, array_size, stage_command, write_sbatch
from ngramprep.ngram_acquire.db.build_path import build_db_path
from ngramprep.ngram_acquire.progress import read_progress

__all__ = ["LexichronApp", "main", "SETTINGS_FILENAME"]

SETTINGS_FILENAME = "lexichron.yaml"


def remember_settings_path(path: Path) -> None:
    try:
        d = _state_dir()
        d.mkdir(parents=True, exist_ok=True)
        (d / "last_settings").write_text(str(path) + "\n", encoding="utf-8")
    except OSError:
        pass


def remember_root(path: Path) -> None:
    try:
        d = _state_dir()
        d.mkdir(parents=True, exist_ok=True)
        (d / "last_root").write_text(str(path) + "\n", encoding="utf-8")
    except OSError:
        pass


def last_root() -> Optional[Path]:
    """The corpus directory the corpus bar showed last time, if it still exists."""
    try:
        text = (_state_dir() / "last_root").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    p = Path(text) if text else None
    return p if p and p.is_dir() else None


def last_settings_path() -> Optional[Path]:
    """The settings file the UI saved most recently, if it still exists."""
    try:
        text = (_state_dir() / "last_settings").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    p = Path(text) if text else None
    return p if p and p.exists() else None


# ---------------------------------------------------------------------------
# Value conversion between widgets and the project dict
# ---------------------------------------------------------------------------

def _to_widget_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(v) for v in value)
    return str(value)


def _from_widget(field: Field, raw: Any) -> Any:
    """Convert a widget value back to a project-file value (None = unset)."""
    if field.kind == "bool":
        return bool(raw)
    if raw is None or raw is Select.NULL:
        return None
    if isinstance(raw, str) and not raw.strip():
        return None
    if field.kind == "optbool":
        return {"yes": True, "no": False}.get(str(raw))
    if field.kind == "int" or (field.kind == "choice" and field.name == "ngram_size"):
        return int(str(raw).strip())
    if field.kind == "float":
        return float(str(raw).strip())
    if field.kind == "grid":
        return list(raw) if isinstance(raw, (list, tuple)) else [raw]
    if field.kind == "list":
        items = [p.strip() for p in str(raw).split(",") if p.strip()]
        conv = []
        for it in items:
            try:
                conv.append(int(it))
            except ValueError:
                conv.append(it)
        return conv
    return str(raw).strip()


def _range_from_widgets(first: str, last: str) -> Optional[List[int]]:
    """Two text boxes -> ``[first, last]`` or None when both are blank."""
    first, last = first.strip(), last.strip()
    if not first and not last:
        return None
    if not first or not last:
        raise ConfigError("file range: fill in both the first and the last file, or neither")
    lo, hi = int(first), int(last)
    if lo < 0 or hi < lo:
        raise ConfigError("file range: first must be >= 0 and not after last")
    return [lo, hi]


class ToggleRow(Horizontal):
    """A setting that takes one or more values from a known set: a button per
    value, lit when selected, plus (for numbers) a box to add another value.
    The last lit button cannot be unlit."""

    class Changed(Message):
        def __init__(self, row: "ToggleRow") -> None:
            super().__init__()
            self.row = row

    def __init__(self, presets: List[str], selected: List[Any], *, numeric: bool, **kwargs) -> None:
        super().__init__(**kwargs)
        self._values: List[str] = [str(v) for v in presets]
        for v in selected:
            if str(v) not in self._values:
                self._values.append(str(v))
        self._selected: List[str] = [str(v) for v in selected] or self._values[:1]
        self.numeric = numeric

    def compose(self) -> ComposeResult:
        for v in self._values:
            yield Button(v, classes="toggle", variant="primary" if v in self._selected else "default",
                         name=v, compact=True)
        if self.numeric:
            yield Input("", placeholder="+ other", classes="toggle-add", compact=True, type="integer")

    @property
    def value(self) -> List[Any]:
        """Selected values in the row's order; numbers as ints."""
        out = [v for v in self._values if v in self._selected]
        return [int(v) for v in out] if self.numeric else out

    def set_value(self, values: List[Any]) -> None:
        self._selected = [str(v) for v in values] or self._values[:1]
        for v in values:
            if str(v) not in self._values:
                self._add_button(str(v))
        self._paint()

    def _add_button(self, v: str) -> None:
        self._values.append(v)
        btn = Button(v, classes="toggle", variant="default", name=v, compact=True)
        adders = self.query(".toggle-add")
        if adders:
            self.mount(btn, before=adders.first())
        else:
            self.mount(btn)

    def _paint(self) -> None:
        for b in self.query(".toggle"):
            b.variant = "primary" if b.name in self._selected else "default"

    @on(Button.Pressed, ".toggle")
    def _toggled(self, event: Button.Pressed) -> None:
        event.stop()
        v = str(event.button.name)
        if v in self._selected:
            if len(self._selected) == 1:
                return                                  # keep at least one
            self._selected.remove(v)
        else:
            self._selected.append(v)
        self._paint()
        self.post_message(self.Changed(self))

    @on(Input.Submitted, ".toggle-add")
    def _added(self, event: Input.Submitted) -> None:
        event.stop()
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        try:
            n = int(text)
        except ValueError:
            return
        if n <= 0:
            return
        v = str(n)
        if v not in self._values:
            self._add_button(v)
        if v not in self._selected:
            self._selected.append(v)
        self._paint()
        self.post_message(self.Changed(self))


class FixedValue(Static):
    """A setting with exactly one valid value: shown as text, still saved."""

    def __init__(self, value: str, **kwargs) -> None:
        super().__init__(value, **kwargs)
        self.value = value


# ---------------------------------------------------------------------------
# The app
# ---------------------------------------------------------------------------

class Meters(Vertical):
    """The figures of a run as labelled tiles in a grid, grouped under
    headings (see progress_view.meters). The widgets are rebuilt only when
    the set of tile labels changes; otherwise headings and values are updated
    in place, so a refresh every few seconds does not flicker. Rebuilding
    mounts finished widgets in one go rather than empty ones to fill later,
    so even then no blank frame is painted."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._shape: tuple = ()
        self._titles: List[Static] = []
        self._tiles: List[List[Static]] = []

    @staticmethod
    def _text(label: str, value: str, note: str) -> str:
        return f"[dim]{label}[/]\n[b]{value}[/]" + (f"  [dim]{note}[/]" if note else "")

    def show(self, groups) -> None:
        shape = tuple(tuple(label for label, _, _ in tiles) for _, tiles in groups)
        if shape != self._shape:
            # Widgets are kept by reference, not id: the old ones are removed
            # asynchronously and would clash with same-named new ones.
            self._shape = shape
            self._titles, self._tiles, nodes = [], [], []
            for title, tiles in groups:
                heading = Static(title, classes="meter-group")
                cells = [Static(self._text(*tile), classes="meter") for tile in tiles]
                self._titles.append(heading)
                self._tiles.append(cells)
                nodes += [heading, Grid(*cells, classes="meter-grid")]
            self.remove_children()
            self.mount(*nodes)
            return
        for heading, cells, (title, tiles) in zip(self._titles, self._tiles, groups):
            heading.update(title)
            for cell, tile in zip(cells, tiles):
                cell.update(self._text(*tile))


def _aligned(value: str, width: int) -> str:
    """A table cell padded to its column's width: numbers (and the "–" of a
    missing one) flush right, labels flush left."""
    return value.rjust(width) if value[:1].isdigit() or value == "–" else value.ljust(width)


class Confirm(ModalScreen[bool]):
    """A question with a safe answer and a risky one; dismisses with True
    when the user takes the risky one."""

    DEFAULT_CSS = """
    Confirm { align: center middle; }
    Confirm > Vertical { width: 72; height: auto; border: thick $error; background: $surface; padding: 1 2; }
    Confirm.warning > Vertical { border: thick $warning; }
    Confirm #confirm-title { text-style: bold; color: $error; }
    Confirm.warning #confirm-title { color: $warning; }
    Confirm #confirm-body { margin: 1 0; }
    Confirm #confirm-buttons { height: auto; align-horizontal: right; }
    Confirm Button { margin-left: 1; }
    """
    BINDINGS = [("escape", "keep", "Back")]

    def __init__(self, title: str, body: str, go: str, keep: str, *, severity: str = "error") -> None:
        super().__init__(classes=severity)
        self.title_text, self.body_text, self.go_label, self.keep_label = title, body, go, keep
        self.severity = severity

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(self.title_text, id="confirm-title")
            yield Static(self.body_text, id="confirm-body")
            with Horizontal(id="confirm-buttons"):
                yield Button(self.keep_label, id="confirm-keep", variant="primary")
                yield Button(self.go_label, id="confirm-go", variant=self.severity)

    @on(Button.Pressed, "#confirm-keep")
    def action_keep(self) -> None:
        self.dismiss(False)

    @on(Button.Pressed, "#confirm-go")
    def _go(self) -> None:
        self.dismiss(True)


class ConfirmOverwrite(Confirm):
    """Asks before a stage deletes earlier output."""

    def __init__(self, risk: OverwriteRisk, action: str) -> None:
        files = f"{risk.nfiles:,} file{'s' if risk.nfiles != 1 else ''}"
        super().__init__(f"This {action} deletes {risk.what}",
                         f"{risk.setting}\n\n{risk.path}\n{risk.size} in {files}\n\n"
                         f"Everything there is lost and the stage starts from nothing. "
                         f"To keep it, {risk.keep_hint} and the stage carries on from what is there.",
                         f"Delete {risk.size} and start over", "Keep it")
        self.risk = risk


class ConfirmMemory(Confirm):
    """Asks before submitting a job whose memory looks too small for its workers."""

    def __init__(self, estimate: MemoryEstimate) -> None:
        super().__init__("This job's memory looks too small",
                         f"{estimate.describe()}\n\nThe estimate is rough (working figures from 5-gram runs), "
                         f"but a job that outgrows its memory is killed after waiting in the queue.",
                         "Submit anyway", "Back to the settings", severity="warning")
        self.estimate = estimate


class ConfirmReplaceSettings(Confirm):
    """Asks before Save replaces a settings file that this session did not
    open: the form was not filled from it, so saving would overwrite it with
    the form's values and drop anything the form does not show."""

    def __init__(self, path: Path) -> None:
        super().__init__("This replaces a settings file you did not open",
                         f"{path}\n\nThe file exists, but this session did not open it, so the form was not "
                         f"filled from it. Saving replaces everything in it with what the form shows; settings "
                         f"it holds that the form does not show are lost. To build on it instead, go back and "
                         f"open it:  lexichron ui {path}",
                         "Replace the file", "Keep the file", severity="warning")
        self.path = path


class NewCorpus(ModalScreen[Optional[Dict[str, Any]]]):
    """Where a corpus that does not exist yet will live: Acquire creates it
    there. Dismisses with the corpus identity, or None."""

    DEFAULT_CSS = """
    NewCorpus { align: center middle; }
    NewCorpus > Vertical { width: 84; height: auto; border: thick $accent; background: $surface; padding: 1 2; }
    NewCorpus #new-title { text-style: bold; color: $accent; }
    NewCorpus #new-note, NewCorpus #new-error { color: $text-muted; margin: 1 0; height: auto; }
    NewCorpus #new-error { color: $warning; }
    NewCorpus .row { height: auto; margin-bottom: 1; }
    NewCorpus .row Label { width: 18; }
    NewCorpus .row Input, NewCorpus .row Select { width: 1fr; }
    NewCorpus #new-buttons { height: auto; align-horizontal: right; }
    NewCorpus Button { margin-left: 1; }
    """
    BINDINGS = [("escape", "cancel", "Back")]

    def __init__(self, root: str) -> None:
        super().__init__()
        self.root = root

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("New corpus", id="new-title")
            yield Static("Acquire creates it: the Google Books n-gram files for this language and size are "
                         "downloaded into <corpus directory>/<release>/<language>/<N>gram_files, and the other "
                         "stages open up as it is built.", id="new-note")
            with Horizontal(classes="row"):
                yield Label("Corpus directory")
                yield Input(value=self.root, id="new-root", compact=True, placeholder="/scratch/<you>/NLP_corpora/Google_Books")
            with Horizontal(classes="row"):
                yield Label("Release")
                yield Static(CHOICES["release"][0], id="new-release")
            with Horizontal(classes="row"):
                yield Label("Language")
                yield Select([(c, c) for c in CHOICES["language"]], allow_blank=True, id="new-language", compact=True)
            with Horizontal(classes="row"):
                yield Label("N-gram size")
                yield Select([(c, c) for c in CHOICES["ngram_size"]], allow_blank=True, id="new-size", compact=True)
            yield Static("", id="new-error")
            with Horizontal(id="new-buttons"):
                yield Button("Cancel", id="new-cancel")
                yield Button("Create", id="new-create", variant="primary")

    @on(Button.Pressed, "#new-cancel")
    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#new-create")
    def _create(self) -> None:
        root = self.query_one("#new-root", Input).value.strip()
        lang = self.query_one("#new-language", Select).value
        size = self.query_one("#new-size", Select).value
        if not root or lang is Select.NULL or size is Select.NULL:
            self.query_one("#new-error", Static).update("Fill in the corpus directory, language and n-gram size.")
            return
        self.dismiss({"db_path_stub": root, "release": CHOICES["release"][0], "language": str(lang),
                      "ngram_size": int(str(size))})


class LexichronApp(App):
    TITLE = f"lexichron {__version__}"
    CSS = """
    #form { height: 1fr; padding: 0 1; scrollbar-size-vertical: 1; }
    .section { text-style: bold; color: $accent; margin-top: 1; }
    .field-help { color: $text-muted; padding-left: 22; margin-bottom: 1; display: none; }
    .show-help .field-help { display: block; }
    .row { height: auto; }
    .row Label { width: 22; }
    /* The corpus sits in its own titled box; a blank line separates it from
       the stage bar, which keeps the panel colour of the tabs below. */
    #corpus-block { height: auto; margin: 0 1; padding: 0 1; border: round $primary; background: $boost; }
    #corpus-bar { height: 1; }
    #corpus-bar #corpus-root { width: 2fr; }
    #corpus-bar #corpus-select { width: 1fr; margin-left: 1; }
    #corpus-bar #corpus-new { margin-left: 1; }
    #corpus-state { height: 1; color: $text-muted; }
    #corpus-fields { display: none; }
    #stage-bar { height: 3; padding: 0 1; margin-top: 1; background: $panel; }
    #stage-bar Label { width: auto; height: 3; content-align: left middle; padding-right: 2; }
    #stage-bar .stage-button { min-width: 14; margin-right: 1; }
    #stage-bar .stage-desc { width: 1fr; height: 3; content-align: left middle; color: $text-muted; padding-left: 1; }
    .stage-box { height: auto; }
    .row Input, .row Select { width: 1fr; }
    .row .range-sep { width: auto; padding: 0 1; color: $text-muted; }
    .row Switch { height: 1; border: none; padding: 0; }
    .row .spacer { width: 1fr; }
    .hint { height: 3; content-align: left middle; color: $text-muted; }
    .row FixedValue { width: 1fr; color: $text-muted; }
    .row .toggle-row { width: 1fr; height: 1; }
    .toggle-row .toggle { min-width: 6; margin-right: 1; }
    .toggle-row .toggle-add { width: 12; }
    .grid-summary { height: auto; padding: 0 1 0 22; color: $text-muted; }
    .row .help-mark { width: 3; color: $text-muted; text-align: center; }
    Collapsible { margin-top: 1; }
    CollapsibleTitle { padding: 0; }
    #dest { padding: 0 1; height: auto; }
    #call { padding: 0 1; }
    #status { color: $warning; padding: 0 1; height: auto; }
    #submit-memory { height: auto; margin: 0 0 1 0; }
    .search-opt { margin-left: 2; }
    #submit-memory.over { color: $warning; }
    .actions { height: auto; padding: 1 1 0 1; }
    .actions Button { margin-right: 2; }
    #runlog { height: 1fr; }
    #jobs { height: auto; max-height: 8; }
    #runs { height: auto; max-height: 8; }
    #inspect-table { height: auto; max-height: 60%; }
    #search-meters { height: auto; }
    #search-meters .meter-group { display: none; }
    #search-grid, #search-pairs, #search-empty { height: auto; padding: 0 1; color: $text-muted; }
    #search-grid { margin: 0 0 1 0; }
    #search-table { height: auto; max-height: 45%; }
    #search-cell { height: auto; padding: 1 1 0 1; text-style: bold; }
    #search-years { height: auto; max-height: 12; }
    #search-legend { margin: 1 0 0 0; }
    #search-legend Static { height: auto; padding: 0 1; color: $text-muted; }
    #search-name { width: 40; }
    #ensemble-name { width: 40; }
    #ensemble-meters { height: auto; }
    #ensemble-meters .meter-group { display: none; }
    #ensemble-empty, #ensemble-note { height: auto; padding: 0 1; color: $text-muted; }
    #ensemble-table { height: auto; max-height: 60%; }
    #ensemble-legend { margin: 1 0 0 0; }
    #ensemble-legend Static { height: auto; padding: 0 1; color: $text-muted; }
    #train-summary { height: auto; padding: 0 1; color: $text-muted; }
    #tuned-cell { margin: 0 0 0 2; }
    #inspect-summary, #inspect-note { height: auto; padding: 0 1; color: $text-muted; }
    #inspect-provenance { height: auto; padding: 0 1; }
    #inspect-detail { height: auto; max-height: 12; padding: 0 1; overflow-y: auto; }
    #inspect-rows { width: 8; }
    .row .inline-label { width: auto; padding: 0 1 0 2; }
    #inspect-layout { width: 16; }
    #run-panel { height: auto; padding: 0 1; border-top: solid $secondary; }
    #run-title { text-style: bold; }
    #run-bar { width: 1fr; margin: 0 0 1 0; }
    #run-bar Bar { width: 1fr; }
    #run-headline, #run-inputs { height: auto; }
    #meters { height: auto; }
    .meter-group { height: auto; color: $text-muted; margin: 1 0 0 0; }
    .meter-grid { layout: grid; grid-size: 4; grid-gutter: 0 1; grid-rows: auto; height: auto; }
    .meter { height: auto; border: round $secondary; padding: 0 1; }
    #run-message { color: $warning; height: auto; }
    #run-log { height: 12; border: round $secondary; scrollbar-size-vertical: 1; }
    .muted { color: $text-muted; }
    """
    BINDINGS = [("ctrl+s", "save", "Save"), ("f1", "toggle_help", "Help"), ("ctrl+q", "quit", "Quit")]

    def __init__(self, project_path: Optional[str | os.PathLike] = None, stage: str = "acquire") -> None:
        super().__init__()
        # The settings file is optional. Without one, the UI reopens the file
        # it saved last time (so closing and reopening keeps the settings and
        # the runs in view); failing that, it lives at <db_path_stub>/lexichron.yaml
        # once that field is filled in, so a user who only wants to download a
        # corpus never has to think about it.
        self.explicit_project: Optional[Path] = Path(project_path).resolve() if project_path else None
        self.reopened = False
        if self.explicit_project is None:
            last = last_settings_path()
            if last:
                self.explicit_project = last.resolve()
                self.reopened = True
        # Every stage's form is built up front; the stage selector shows one.
        self.stage = stage
        self.stage_funcs = {name: _resolve(dotted) for name, (dotted, _) in STAGES.items()}
        self.stage_sections: Dict[str, Section] = {}
        corpus = Section("corpus")
        seen = set()
        for name, func in self.stage_funcs.items():
            corp, sec = stage_sections(func, name)
            self.stage_sections[name] = sec
            for f in corp.fields:               # union of the stages' corpus settings
                if f.name not in seen:
                    seen.add(f.name)
                    corpus.fields.append(f)
                elif f.required:
                    next(c for c in corpus.fields if c.name == f.name).required = True
        self.corpus_section = corpus
        self.sections: List[Section] = [corpus] + list(self.stage_sections.values()) + [SLURM_FIELDS]
        self.config: Dict[str, Any] = {}
        # Settings files this session opened or wrote: Save replaces any other
        # file on disk only after asking (see ConfirmReplaceSettings).
        self._known_settings: set = set()
        if self.explicit_project and self.explicit_project.exists():
            self.config = load_project(self.explicit_project)
            self._known_settings.add(self.explicit_project.resolve())
        self.loaded_dir: Optional[Path] = self._corpus_dir(self.config)
        stub = (self.config.get("corpus") or {}).get("db_path_stub")
        self._root_initial = str(stub) if stub else str(last_root() or "")
        self._corpus_infos: Dict[str, CorpusInfo] = {}
        self._listed_dir: Optional[Path] = None
        self._selecting = False
        self._populating = False
        self._bar_ready = False
        # The slurm section as saved (self.config follows the form as it is
        # typed): the Submit tab's sources line is read against this.
        self._saved_slurm: Dict[str, Any] = dict(self.config.get("slurm") or {})
        self.bridge: Optional[SlurmRunner] = slurm_backend()
        self._status: Dict[str, Any] = {"jobs": None, "jobs_error": "", "docs": [], "selected": None, "log_lines": []}
        self._gathering = False
        self._squeue_at = float("-inf")
        self.proc: Optional[subprocess.Popen] = None

    @property
    def project_path(self) -> Optional[Path]:
        """Where the settings file is (or will be) saved: by default beside
        the database it describes, <stub>/<release>/<language>/<n>gram_files/
        lexichron.yaml. A file that was opened explicitly (command-line
        argument, or reopened from last time) stays the target until the
        corpus it describes is changed, at which point the file follows."""
        corpus_dir = self._corpus_dir(self.config)
        if self.explicit_project and (corpus_dir is None or corpus_dir == self.loaded_dir
                                      or self.explicit_project.parent == corpus_dir):
            return self.explicit_project
        return corpus_dir / SETTINGS_FILENAME if corpus_dir else None

    @property
    def func(self):
        return self.stage_funcs[self.stage]

    def _corpus_dir(self, config: Dict[str, Any]) -> Optional[Path]:
        """The directory of the database these settings describe, or None
        until the corpus directory, release, language and n-gram size are set."""
        corpus = config.get("corpus") or {}
        stage = config.get(self.stage) or {}
        stub, release, lang = corpus.get("db_path_stub"), corpus.get("release"), corpus.get("language")
        n = corpus.get("ngram_size") or stage.get("ngram_size")
        if not (stub and release and lang and n):
            return None
        root = Path(str(stub)).expanduser().resolve()
        return Path(build_db_path(str(root), int(n), str(release), str(lang))).parent

    def _require_project_path(self) -> Path:
        p = self.project_path
        if p is None:
            raise ConfigError("fill in the corpus directory, language and n-gram size first")
        return p

    # -- layout -----------------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Header()
        # The corpus is chosen first, above everything: its settings file,
        # runs, searches and ensembles are what the tabs show, and the stage
        # buttons light up as the corpus is built. Acquire creates a corpus
        # (New corpus); the identity fields themselves stay on the form,
        # hidden, so that saving and validation read them as before.
        with Vertical(id="corpus-block"):
            with Horizontal(id="corpus-bar"):
                yield Input(value=self._root_initial, id="corpus-root", compact=True,
                            placeholder="corpus directory, e.g. /scratch/<you>/NLP_corpora/Google_Books")
                yield Select([], allow_blank=True, id="corpus-select", compact=True, prompt="(corpora found there)")
                yield Button("New corpus", id="corpus-new", compact=True)
            yield Static("", id="corpus-state")
        # The stage is chosen once, above the tabs: Settings, Run and Submit
        # act on it, and it stays in view whichever tab is open.
        with Horizontal(id="stage-bar"):
            yield Label("Stage")
            for name, (_, desc) in STAGES.items():
                b = Button(name.capitalize(), id=f"stage-btn-{name}", classes="stage-button")
                b.tooltip = desc
                yield b
            yield Static("", id="stage-desc", classes="stage-desc")
        with TabbedContent(initial="tab-project"):
            with TabPane("Settings", id="tab-project"):
                yield Static("", id="dest")
                yield Static("", id="status")
                with VerticalScroll(id="form"):
                    yield Static("Corpus", id="corpus-line", classes="section")
                    with Vertical(id="corpus-fields"):
                        for f in self.corpus_section.fields:
                            if not f.advanced:
                                yield from self._field_widgets("corpus", f)
                    for name, sec in self.stage_sections.items():
                        with Vertical(id=f"stage-{name}", classes="stage-box"):
                            yield Static(name.capitalize(), classes="section")
                            for f in sec.fields:
                                if not f.advanced:
                                    yield from self._field_widgets(name, f)
                            if name == "tune":
                                yield Static("", id="grid-summary", classes="grid-summary")
                            if name == "train":
                                with Horizontal(classes="row"):
                                    yield Label("From search")
                                    yield Select([], allow_blank=True, id="tuned-search", compact=True,
                                                 prompt="(searches of this corpus)")
                                    yield Button("Use its suggested cell", id="tuned-cell", compact=True)
                                yield Static("", id="train-summary")
                    with Collapsible(title="Advanced", collapsed=True, id="advanced"):
                        for f in self.corpus_section.fields:
                            if f.advanced:
                                yield from self._field_widgets("corpus", f)
                        for name, sec in self.stage_sections.items():
                            with Vertical(id=f"adv-{name}", classes="stage-box"):
                                for f in sec.fields:
                                    if f.advanced:
                                        yield from self._field_widgets(name, f)
                    with Collapsible(title="Resolved call (what --dry-run prints)", collapsed=True,
                                     id="call-box"):
                        yield Static("", id="call")
                with Horizontal(classes="actions"):
                    yield Button("Save settings", id="save", variant="primary")
                    yield Static("  F1 shows help for every setting; hover a ? for one.", classes="hint")
            with TabPane("Run", id="tab-run"):
                with Horizontal(classes="actions"):
                    yield Button("Run here", id="run", variant="success")
                    yield Button("Stop", id="stop", variant="error")
                yield RichLog(id="runlog", wrap=True, highlight=False, markup=False)
            with TabPane("Submit", id="tab-submit"):
                with VerticalScroll():
                    yield Static("Slurm resources", classes="section")
                    for f in SLURM_FIELDS.fields:
                        yield from self._field_widgets("slurm", f)
                    yield Static("", id="submit-sources", classes="muted")
                    with Horizontal(classes="actions"):
                        yield Button("Write batch script", id="write-sbatch")
                        yield Button("Submit to Slurm", id="submit", variant="primary")
                        yield Button("Use account as my default", id="slurm-defaults")
                    yield Static("", id="submit-command", classes="muted")
                    yield Static("", id="submit-memory", classes="muted")
                    yield Static("", id="submit-status")
            with TabPane("Progress", id="tab-jobs"):
                with VerticalScroll():
                    yield Static("Runs (select one to see its progress)", classes="section")
                    yield DataTable(id="runs")
                    with Vertical(id="run-panel"):
                        yield Static("", id="run-title")
                        yield ProgressBar(total=None, show_eta=False, id="run-bar")
                        yield Static("", id="run-headline")
                        yield Static("", id="run-inputs", classes="muted")
                        yield Static("", id="run-message")
                        yield Meters(id="meters")
                        with Collapsible(title="Log", collapsed=True, id="run-log-box"):
                            yield RichLog(id="run-log", wrap=False, highlight=False, markup=False)
                    yield Static("Slurm jobs", classes="section")
                    yield DataTable(id="jobs")
                    with Horizontal(classes="actions"):
                        yield Button("Refresh", id="refresh-jobs")
                        yield Button("Cancel selected job", id="cancel-job", variant="error")
                        yield Button("Clear finished runs", id="clear-runs")
            with TabPane("Inspect", id="tab-inspect"):
                # the tune stage: a search's scores, one row per cell
                with Vertical(id="inspect-search"):
                    with Horizontal(classes="row"):
                        yield Label("Search")
                        yield Select([], allow_blank=True, id="search-name", compact=True,
                                     prompt="(searches of this corpus)")
                        yield Button("Reload", id="search-reload", compact=True)
                        yield Label("Columns", classes="search-opt")
                        yield Select([("All", "all"), ("Benchmarks", "benchmarks"),
                                      ("Neighbourhoods", "neighbourhoods"), ("Valence axis", "axis")],
                                     value="all", allow_blank=False, id="search-view", compact=True)
                        yield Label("± spreads", classes="search-opt")
                        yield Switch(value=True, id="search-spreads")
                    yield Static("", id="search-empty")
                    yield Meters(id="search-meters")
                    yield Static("", id="search-grid")
                    yield DataTable(id="search-table", zebra_stripes=True)
                    yield Static("", id="search-cell")
                    yield DataTable(id="search-years", zebra_stripes=True)
                    yield Static("", id="search-pairs")
                    with Collapsible(title="What the columns mean", collapsed=True, id="search-legend"):
                        yield Static("", id="search-legend-text")
                # the train stage: an ensemble's years
                with Vertical(id="inspect-ensemble"):
                    with Horizontal(classes="row"):
                        yield Label("Ensemble")
                        yield Select([], allow_blank=True, id="ensemble-name", compact=True,
                                     prompt="(ensembles of this corpus)")
                        yield Button("Reload", id="ensemble-reload", compact=True)
                        yield Label("Columns", classes="search-opt")
                        yield Select([("All", "all"), ("Benchmarks", "benchmarks"),
                                      ("Neighbourhoods", "neighbourhoods"), ("Valence axis", "axis")],
                                     value="all", allow_blank=False, id="ensemble-view", compact=True)
                    yield Static("", id="ensemble-empty")
                    yield Meters(id="ensemble-meters")
                    yield DataTable(id="ensemble-table", zebra_stripes=True)
                    yield Static("", id="ensemble-note")
                    with Collapsible(title="What the columns mean", collapsed=True, id="ensemble-legend"):
                        yield Static(ENSEMBLE_LEGEND, id="ensemble-legend-text")
                # the other stages: records of a database or vocabulary file.
                # The corpus is chosen in the bar above; this picks which of
                # its databases or files to open.
                with Vertical(id="inspect-generic"):
                    with Horizontal(classes="row"):
                        yield Label("Database")
                        yield Select([], allow_blank=True, id="inspect-db", compact=True,
                                     prompt="(which database or file)")
                    with Horizontal(classes="row"):
                        yield Label("Path")
                        yield Input("", id="inspect-path", compact=True,
                                    placeholder="any RocksDB directory or vocabulary file")
                    yield Static("", id="inspect-summary")
                    with Collapsible(title="How it was made", collapsed=True, id="inspect-provenance-box"):
                        yield Static("", id="inspect-provenance")
                    with Horizontal(classes="row"):
                        yield Label("Key")
                        yield Input("", id="inspect-key", compact=True,
                                    placeholder="[2011] nuclear for a pivoted database; the n-gram otherwise")
                        yield Label("Rows", classes="inline-label")
                        yield Input("20", id="inspect-rows", compact=True, type="integer")
                        yield Label("Keys", classes="inline-label")
                        yield Select([(l, l) for l in LAYOUTS], value="auto", allow_blank=False,
                                     id="inspect-layout", compact=True)
                    with Horizontal(classes="actions"):
                        yield Button("First rows", id="inspect-head", variant="primary")
                        yield Button("From key", id="inspect-from-key")
                        yield Button("With prefix", id="inspect-prefix")
                    yield Static("", id="inspect-note")
                    yield DataTable(id="inspect-table")
                    yield Static("", id="inspect-detail")
        yield Footer()

    def _field_widgets(self, section: str, f: Field):
        wid = f"f-{section}-{f.name}"
        current = (self.config.get(section) or {}).get(f.name, f.default)
        label = f.title + (" *" if f.required else "")
        # the row and its help line carry ids so a field can be shown or hidden by stage
        with Horizontal(classes="row", id=f"row-{section}-{f.name}"):
            yield Label(label)
            if f.kind == "bool":
                w = Switch(value=bool(current), id=wid)
                w.tooltip = f.help or None
                yield w
                w = Static("", classes="spacer")
            elif f.kind == "optbool":
                val = Select.NULL if current is None else ("yes" if current else "no")
                w = Select([("yes", "yes"), ("no", "no")], value=val, allow_blank=True, id=wid, compact=True,
                           prompt="(default)")
            elif f.kind == "grid":
                selected = current if isinstance(current, (list, tuple)) else ([current] if current is not None else [])
                w = ToggleRow(list(f.choices or []), list(selected), numeric=AXIS_BY_NAME[f.name].kind == "int",
                              id=wid, classes="toggle-row")
            elif f.kind == "choice" and f.choices and len(f.choices) == 1:
                w = FixedValue(f.choices[0], id=wid)   # only one valid value
            elif f.kind == "choice" and f.choices:
                opts = [(c, c) for c in f.choices]
                val = str(current) if current is not None else Select.NULL
                if val is not Select.NULL and val not in f.choices:
                    opts.append((val, val))
                w = Select(opts, value=val, allow_blank=True, id=wid, compact=True)
            elif f.kind == "range":
                lo, hi = ("", "")
                if isinstance(current, (list, tuple)) and len(current) == 2:
                    lo, hi = str(current[0]), str(current[1])
                w = Input(value=lo, id=wid, compact=True, placeholder="first (blank = all)", type="integer")
                w.tooltip = f.help or None
                yield w
                yield Static("to", classes="range-sep")
                w = Input(value=hi, id=f"{wid}-end", compact=True, placeholder="last", type="integer")
            else:
                hint = {"path": "path", "list": "comma-separated", "int": "number"}.get(f.kind, f.kind)
                w = Input(value=_to_widget_text(current), id=wid, compact=True,
                          placeholder=f"({hint}{', optional' if not f.required else ''})")
            w.tooltip = f.help or None
            yield w
            mark = Static("?" if f.help else "", classes="help-mark")
            mark.tooltip = f.help or None
            yield mark
        if f.help:
            yield Static(f.help, classes="field-help", id=f"help-{section}-{f.name}")

    def action_toggle_help(self) -> None:
        """F1: show or hide the help line under every setting."""
        self.screen.toggle_class("show-help")

    def _show_stage(self, stage: str) -> None:
        """Make ``stage`` the one the form, Run and Submit act on: its button
        lit, its settings shown, its Slurm resources on the Submit tab."""
        if getattr(self, "_submit_ready", False) and stage != self.stage:
            self._stash_slurm()
        self.stage = stage
        for name in self.stage_sections:
            self.query_one(f"#stage-{name}").display = name == stage
            self.query_one(f"#adv-{name}").display = name == stage
        for name in STAGES:
            self.query_one(f"#stage-btn-{name}", Button).variant = "primary" if name == stage else "default"
        self.query_one("#stage-desc", Static).update(STAGES[stage][1])
        # Jobs (a job array) only means something for the stages that split their years
        for wid in ("#row-slurm-jobs", "#help-slurm-jobs"):
            self.query_one(wid).display = stage in ARRAY_STAGES
        self._submit_ready = True
        self._populate_slurm()
        self.query_one("#inspect-search").display = stage == "tune"
        self.query_one("#inspect-ensemble").display = stage == "train"
        self.query_one("#inspect-generic").display = stage not in ("tune", "train")
        if stage == "tune":
            self._refresh_searches()
        if stage == "train":
            self._refresh_ensembles()
            self._refresh_tuned_searches()

    @on(Button.Pressed, ".stage-button")
    def _stage_button_pressed(self, event: Button.Pressed) -> None:
        stage = str(event.button.id)[len("stage-btn-"):]
        if stage != self.stage:
            self.select_stage(stage)

    def select_stage(self, stage: str) -> None:
        self._show_stage(stage)
        self._refresh_preview()

    def on_mount(self) -> None:
        self.query_one("#corpus-block").border_title = "Corpus"
        self._show_stage(self.stage)
        self.query_one(f"#stage-btn-{self.stage}", Button).focus()   # initial focus on the lit button
        jobs = self.query_one("#jobs", DataTable)
        jobs.add_columns("Job", "Name", "State", "Elapsed", "Limit", "Nodes", "Reason")
        jobs.cursor_type = "row"
        runs = self.query_one("#runs", DataTable)
        self._run_columns = runs.add_columns("Run", "State", "Done", "Records", "Rate", "Elapsed", "Job")
        runs.cursor_type = "row"
        self.query_one("#inspect-table", DataTable).add_columns("Key", "Value")
        self.query_one("#inspect-table", DataTable).cursor_type = "row"
        self.query_one("#search-table", DataTable).cursor_type = "row"
        self.query_one("#search-years", DataTable).cursor_type = "none"
        self._inspect_dir: Optional[Path] = None
        self._search: Optional[Dict[str, Any]] = None
        self._ensemble: Optional[Dict[str, Any]] = None
        self._ensemble_at = 0.0
        self._search_sort: Tuple[Optional[str], bool] = (None, False)
        self.selected_run: Optional[Path] = None
        self._log_shown: Optional[tuple] = None
        self.queued_jobs: Optional[set] = None
        self._bar_ready = True
        self._refresh_corpus_list()
        self._refresh_preview()
        self.refresh_jobs()
        self.set_interval(5, self.refresh_jobs)
        if not self.bridge or not self.bridge.available():
            self.query_one("#submit-status", Static).update(self._no_slurm_note())

    def _no_slurm_note(self) -> str:
        """Why 'Submit' is unavailable, so the fix is clear from the note alone."""
        if os.environ.get("APPTAINER_CONTAINER") or os.environ.get("SINGULARITY_CONTAINER"):
            why = ("this UI is running inside the image without the host helper. Start it with the "
                   "lexichron-ui command; if you meant a development install, 'which lexichron' is "
                   "naming the image launcher instead of your environment's command")
        elif self.bridge is None:
            why = "sbatch is not on this machine's PATH and LEXICHRON_BRIDGE_DIR is not set"
        else:
            why = f"the host helper in {getattr(self.bridge, 'dir', '?')} is not running"
        return (f"No Slurm here: {why}. 'Submit' and the job table need it; "
                "'Write batch script' still works.")

    # -- form -> config -----------------------------------------------------------
    def _slurm_form(self) -> Dict[str, Any]:
        """The Submit tab's values as typed (invalid ones raise ConfigError)."""
        vals: Dict[str, Any] = {}
        for f in SLURM_FIELDS.fields:
            try:
                v = _from_widget(f, self.query_one(f"#f-slurm-{f.name}").value)
            except ValueError:
                raise ConfigError(f"{f.title}: not a valid {f.kind}")
            if v is not None:
                vals[f.name] = v
        return vals

    def _slurm_resolved(self) -> Dict[str, Any]:
        """The Slurm settings a job of the current stage would get, from the
        saved settings and the user's defaults (not the form)."""
        return resolve_slurm(self.stage, self.config.get("slurm"))[0]

    def _collect(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {}
        for sec in self.sections:
            vals: Dict[str, Any] = {}
            if sec.name == "slurm":
                # the Submit tab's values become the stage's own block; the
                # account and partition stay out while they match the user's defaults
                layered = stage_override(self.config.get("slurm"), self.stage, self._slurm_form())
                if layered:
                    cfg["slurm"] = layered
                continue
            for f in sec.fields:
                w = self.query_one(f"#f-{sec.name}-{f.name}")
                try:
                    if f.kind == "range":
                        v = _range_from_widgets(w.value, self.query_one(f"#f-{sec.name}-{f.name}-end").value)
                    else:
                        v = _from_widget(f, w.value)
                except ConfigError:
                    raise
                except ValueError:
                    raise ConfigError(f"{f.title}: not a valid {f.kind}")
                if v is None or (f.kind == "bool" and v == f.default) or v == f.default and not f.required:
                    continue
                vals[f.name] = v
            if vals:
                cfg[sec.name] = vals
        return cfg

    def _refresh_preview(self) -> None:
        call = self.query_one("#call", Static)
        status = self.query_one("#status", Static)
        try:
            cfg = self._collect()
            self.config = cfg
        except ConfigError as exc:
            status.update(str(exc))
            return
        self._refresh_inspect_candidates()
        if self._bar_ready:
            if self._corpus_dir(cfg) != self._listed_dir:
                self._refresh_corpus_list()
            self._refresh_corpus_state()
        dest = self.project_path
        self.query_one("#dest", Static).update(
            f"Settings file: {dest if dest else '(fill in the corpus directory, language and n-gram size)'}"
            + ("  (reopened from last time)" if self.reopened else ""))
        self._refresh_command_line(dest)
        try:
            kwargs = build_call(self.func, {k: v for k, v in cfg.items() if k in ("corpus", self.stage)},
                                self.stage)
        except ConfigError as exc:
            call.update("(fill in the required settings)")
            status.update(self._friendly(str(exc)))
            return
        call.update(_format_call(self.func, kwargs))
        status.update("")
        self._refresh_memory_line()
        if self.stage == "tune":
            self._refresh_grid_summary(kwargs)
        if self.stage == "train":
            self._refresh_train_summary(kwargs)

    def _refresh_grid_summary(self, kwargs: Dict[str, Any]) -> None:
        """Under the tune form: what the grid amounts to, and whether the
        parallelism fits the job's CPUs."""
        from lexichron.stages.tune import sample_years_in
        widget = self.query_one("#grid-summary", Static)
        try:
            grid = HyperGrid.from_settings(kwargs)
            years = kwargs.get("sample_years") or sample_years_in(tuple(kwargs.get("years", (1968, 2019))),
                                                                 int(kwargs.get("year_pairs", 4)))
            per = int(kwargs.get("seeds", 2)) + (2 if kwargs.get("split_half", True) else 0)
            models = len(grid) * len(years) * per
            cpus = int(self._slurm_resolved().get("cpus") or kwargs.get("workers") or os.cpu_count() or 1)
            threads = max(1, int(kwargs.get("workers_per_model", 4)))
            at_once = max(1, cpus // threads)
            text = (f"Grid: {grid.describe()}\n"
                    f"x {len(years)} years ({', '.join(str(y) for y in years)}) x {per} models each "
                    f"({kwargs.get('seeds', 2)} seed{'s' if int(kwargs.get('seeds', 2)) != 1 else ''}"
                    f"{' + 2 halves' if kwargs.get('split_half', True) else ''}) = {models} models; "
                    f"{at_once} at a time x {threads} threads on {cpus} CPUs")
            if cpus % threads:
                text += f"  (CPUs are not a multiple of the threads per model; {cpus - at_once * threads} idle)"
            text += self._array_note(sorted(set(int(y) for y in years)), len(grid) * per)
        except (ValueError, TypeError) as exc:
            text = f"Grid: {exc}"
        widget.update(text)

    def _refresh_train_summary(self, kwargs: Dict[str, Any]) -> None:
        """Under the train form: the ensemble's size (every year of the study
        is assumed to have data), the parallelism, and the split over jobs."""
        widget = self.query_one("#train-summary", Static)
        try:
            first, last = (int(v) for v in kwargs.get("years", (1968, 2019)))
            years = list(range(first, last + 1))
            corpora = int(kwargs.get("corpus_replicates", 20)) if kwargs.get("resampling", "poisson") != "none" else 1
            seeds = int(kwargs.get("seeds_per_corpus", 2))
            per_year = corpora * seeds
            cpus = int(self._slurm_resolved().get("cpus") or kwargs.get("workers") or os.cpu_count() or 1)
            threads = max(1, int(kwargs.get("workers_per_model", 4)))
            at_once = max(1, cpus // threads)
            text = (f"Ensemble: {len(years)} years ({first}-{last}) x {corpora} corpus x {seeds} seed replicates "
                    f"= {per_year} models per year, {len(years) * per_year} in all; "
                    f"{at_once} at a time x {threads} threads on {cpus} CPUs")
            text += self._array_note(years, per_year)
        except (ValueError, TypeError) as exc:
            text = f"Ensemble: {exc}"
        widget.update(text)

    def _refresh_tuned_searches(self) -> None:
        """The searches a train run can take its cell from."""
        corpus_dir = self._corpus_dir(self.config)
        found = searches(str(corpus_dir) if corpus_dir else None)
        sel = self.query_one("#tuned-search", Select)
        current = sel.value if sel.value is not Select.NULL else None
        sel.set_options(found)
        if found and not any(path == current for _, path in found):
            sel.value = found[0][1]

    @on(Button.Pressed, "#tuned-cell")
    def _tuned_cell_pressed(self) -> None:
        """Copy a search's suggested cell into the train form's six fields."""
        sel = self.query_one("#tuned-search", Select)
        if sel.value is Select.NULL:
            self.notify("No tuning search to take a cell from", severity="warning")
            return
        search = read_search(str(sel.value))
        stem = search.get("best_cell")
        if not stem:
            self.notify(f"{search.get('name')}: no suggested cell yet", severity="warning")
            return
        from lexichron.training.grid import Cell
        cell = Cell.from_stem(stem)
        for key, value in cell.as_dict().items():
            w = self.query_one(f"#f-train-{key}")
            if isinstance(w, Select):
                w.value = str(value)
            else:
                w.value = str(value)
        self.notify(f"Cell {stem} from {search.get('name')} ({search.get('best_rule', '')[:60]}…)")
        self._refresh_preview()

    # -- Inspect tab, train stage: an ensemble's years -----------------------
    def _refresh_ensembles(self) -> None:
        corpus_dir = self._corpus_dir(self.config)
        found = ensembles(str(corpus_dir) if corpus_dir else None)
        sel = self.query_one("#ensemble-name", Select)
        current = sel.value if sel.value is not Select.NULL else None
        sel.set_options(found)
        wanted = (self.config.get("train") or {}).get("name")
        choice = next((path for name, path in found if name == wanted), None) or \
            (current if any(path == current for _, path in found) else None) or (found[0][1] if found else None)
        if choice is None:
            self._show_ensemble(None)
            return
        if sel.value != choice:
            sel.value = choice
        else:
            self._load_ensemble(choice)

    @on(Select.Changed, "#ensemble-name")
    def _ensemble_chosen(self, event: Select.Changed) -> None:
        if event.value is not Select.NULL:
            self._load_ensemble(str(event.value))

    @on(Button.Pressed, "#ensemble-reload")
    def _ensemble_reload_pressed(self) -> None:
        self._refresh_ensembles()

    @on(Select.Changed, "#ensemble-view")
    def _ensemble_view_changed(self) -> None:
        if self._ensemble is not None:
            self._show_ensemble(self._ensemble)

    @work(thread=True, group="ensemble", exclusive=True)
    def _load_ensemble(self, path: str) -> None:
        self._ensemble_at = time.monotonic()
        ens = read_ensemble(path)
        self.call_from_thread(self._show_ensemble, ens)

    ENSEMBLE_EVERY_S = 30.0      # while a train run is going, its year records are reread this often

    def _follow_running_ensemble(self, docs: List[tuple]) -> None:
        """A train run rewrites year records as years complete: reread the
        shown ensemble now and then while one is running."""
        if self.stage != "train" or not self._ensemble or self._ensemble.get("error"):
            return
        running = any(doc and doc.get("stage") == "train" and doc.get("state") == "running" for _, doc in docs)
        if running and time.monotonic() - self._ensemble_at >= self.ENSEMBLE_EVERY_S:
            self._load_ensemble(self._ensemble["path"])

    def _show_ensemble(self, ens: Optional[Dict[str, Any]]) -> None:
        self._ensemble = ens
        table = self.query_one("#ensemble-table", DataTable)
        table.clear(columns=True)
        empty = self.query_one("#ensemble-empty", Static)
        has = ens is not None and not ens.get("error")
        for wid in ("#ensemble-meters", "#ensemble-table", "#ensemble-note", "#ensemble-legend"):
            self.query_one(wid).display = has
        if not has:
            empty.update(ens["error"] if ens else
                         "No model ensemble in this corpus directory yet: run the train stage, and its years "
                         "appear here as they finish.")
            empty.display = True
            return
        empty.display = False
        view = self.query_one("#ensemble-view", Select).value
        view = str(view) if view is not Select.NULL else "all"
        self.query_one("#ensemble-meters", Meters).show([("", ensemble_overview(ens)), ensemble_profile(ens, view)])
        headings, rows = ensemble_table(ens, view)
        for h in headings:
            table.add_column(h, key=h)
        widths = [max([len(h)] + [len(r[i]) for r in rows]) for i, h in enumerate(headings)]
        for row in rows:
            table.add_row(*[Text(_aligned(v, widths[i])) for i, v in enumerate(row)], key=row[0])
        note = "" if rows else "(no year finished yet)"
        if ens.get("provenance"):
            note = (note + "\n" if note else "") + "\n".join(format_provenance(ens["provenance"])[:1])
        self.query_one("#ensemble-note", Static).update(note)

    def _array_note(self, years: List[int], models_per_year: int) -> str:
        """How the Submit tab's Jobs setting would split the years."""
        from lexichron.arrays import ArrayTask, task_share
        jobs = array_size(self.stage, self._slurm_resolved())
        if jobs <= 1:
            return ""
        shares = [task_share(years, ArrayTask(i, jobs)) for i in range(jobs)]
        busy = [s for s in shares if s]
        most = max(len(s) for s in shares)
        note = (f"\nSplit over {jobs} jobs (Submit: Jobs): each takes every {jobs}th year, "
                f"up to {most} year{'s' if most != 1 else ''} and {most * models_per_year} models")
        if len(busy) < jobs:
            note += f"; only {len(busy)} of the {jobs} would have a year to do (there are {len(years)})"
        return note

    def _friendly(self, message: str) -> str:
        """Replace ``section.key`` names in a config error with the form's labels."""
        for sec in self.sections:
            for f in sec.fields:
                message = message.replace(f"{sec.name}.{f.name}", f.title)
        return message

    @on(Input.Changed)
    @on(Select.Changed)
    @on(Switch.Changed)
    @on(ToggleRow.Changed)
    def _on_any_change(self, _event) -> None:
        if not self._populating:
            self._refresh_preview()

    # -- the corpus bar -------------------------------------------------------------
    def _current_info(self) -> Optional[CorpusInfo]:
        c = self.config.get("corpus") or {}
        n = c.get("ngram_size") or (self.config.get(self.stage) or {}).get("ngram_size")
        if not (c.get("db_path_stub") and c.get("release") and c.get("language") and n):
            return None
        try:
            return corpus_info(str(c["db_path_stub"]), str(c["release"]), str(c["language"]), int(n))
        except (OSError, ValueError):
            return None

    def _refresh_corpus_list(self) -> None:
        """The corpora under the root on the bar, the current one selected."""
        root_text = self.query_one("#corpus-root", Input).value.strip()
        infos = scan_corpora(root_text) if root_text else []
        self._corpus_infos = {str(i.dir): i for i in infos}
        options = [(i.label, str(i.dir)) for i in infos]
        current = self._corpus_dir(self.config)
        if current and str(current) not in self._corpus_infos:
            c = self.config.get("corpus") or {}
            n = c.get("ngram_size") or (self.config.get(self.stage) or {}).get("ngram_size")
            options.insert(0, (f"{c.get('language')} · {c.get('release')} · {n}-grams", str(current)))
        sel = self.query_one("#corpus-select", Select)
        self._selecting = True
        try:
            sel.set_options(options)
            if current and any(v == str(current) for _, v in options):
                sel.value = str(current)
        finally:
            self._selecting = False
        self._listed_dir = current
        if root_text and Path(root_text).expanduser().is_dir():
            remember_root(Path(root_text).expanduser())

    def _refresh_corpus_state(self) -> None:
        """The line under the bar, the corpus line on the form, and which
        stage buttons are live: a stage needs what the one before it made."""
        info = self._current_info()
        if info is None:
            state = "No corpus chosen: pick one above, or New corpus to acquire one."
            line = "Corpus: none chosen (use the corpus bar above)"
        else:
            where = f"settings {info.settings}" if info.settings else "no settings file yet (Save writes one)"
            state = f"{info.label}: {info.state} · {where}"
            line = f"Corpus: {info.label}  ({info.dir})"
        self.query_one("#corpus-state", Static).update(state)
        self.query_one("#corpus-line", Static).update(line)
        pre = stage_prerequisites(info)
        for name, (_, desc) in STAGES.items():
            btn = self.query_one(f"#stage-btn-{name}", Button)
            btn.disabled = bool(pre.get(name))
            btn.tooltip = pre.get(name) or desc

    @on(Input.Changed, "#corpus-root")
    def _corpus_root_changed(self) -> None:
        if self._bar_ready:
            self._refresh_corpus_list()

    @on(Select.Changed, "#corpus-select")
    def _corpus_chosen(self, event: Select.Changed) -> None:
        if self._selecting or event.value is Select.NULL:
            return
        chosen = str(event.value)
        current = self._corpus_dir(self.config)
        if current and chosen == str(current):
            return
        info = self._corpus_infos.get(chosen)
        if info is not None:
            self._open_corpus(info)

    @on(Button.Pressed, "#corpus-new")
    def _corpus_new_pressed(self) -> None:
        root = self.query_one("#corpus-root", Input).value.strip()
        self.push_screen(NewCorpus(root), lambda d: self._start_new_corpus(d) if d else None)

    def _start_new_corpus(self, ident: Dict[str, Any]) -> None:
        info = corpus_info(ident["db_path_stub"], ident["release"], ident["language"], ident["ngram_size"])
        root_input = self.query_one("#corpus-root", Input)
        if root_input.value.strip() != ident["db_path_stub"]:
            root_input.value = ident["db_path_stub"]
        existing = info.settings or info.database or info.pivoted
        self._open_corpus(info, stage=None if existing else "acquire")
        if existing:
            self.notify(f"{info.label} exists there already; opened it")
        else:
            self.notify(f"New corpus {info.label}: fill in the Acquire settings, then Run or Submit")

    def _open_corpus(self, info: CorpusInfo, stage: Optional[str] = None) -> None:
        """Make ``info`` the corpus everything acts on: its settings file
        opened if it has one, else a fresh form with its identity filled in."""
        if info.settings:
            try:
                cfg = load_project(info.settings)
            except ConfigError as exc:
                self.notify(f"{info.settings}: {exc}", severity="error")
                return
            self.explicit_project = info.settings.resolve()
            self._known_settings.add(self.explicit_project)
            remember_settings_path(info.settings)
        else:
            cfg = {"corpus": {"db_path_stub": str(info.root), "release": info.release,
                              "language": info.language, "ngram_size": info.ngram_size}}
            self.explicit_project = None
        self.reopened = False
        self.config = cfg
        self.loaded_dir = info.dir
        self._saved_slurm = dict(cfg.get("slurm") or {})
        self._populate_form(cfg)
        pre = stage_prerequisites(info)
        wanted = stage or self.stage
        if pre.get(wanted):
            wanted = next(name for name in STAGES if not pre.get(name))
        self._show_stage(wanted)
        self._refresh_preview()

    def _populate_form(self, cfg: Dict[str, Any]) -> None:
        """Every field widget from ``cfg`` (the Submit tab from the stage's
        resolved Slurm settings)."""
        self._populating = True
        try:
            for sec in self.sections:
                if sec.name == "slurm":
                    continue
                vals = cfg.get(sec.name) or {}
                for f in sec.fields:
                    self._set_field_widget(sec.name, f, vals.get(f.name, f.default))
        finally:
            self._populating = False
        if getattr(self, "_submit_ready", False):
            self._populate_slurm()

    def _set_field_widget(self, section: str, f: Field, value: Any) -> None:
        wid = f"#f-{section}-{f.name}"
        w = self.query_one(wid)
        if isinstance(w, FixedValue):
            return
        if f.kind == "bool":
            w.value = bool(value)
        elif f.kind == "optbool":
            w.value = Select.NULL if value is None else ("yes" if value else "no")
        elif f.kind == "grid":
            chosen = list(value) if isinstance(value, (list, tuple)) else ([value] if value is not None else [])
            w.set_value(chosen or list(f.default or []))
        elif f.kind == "choice" and f.choices:
            val = str(value) if value is not None else Select.NULL
            opts = [(c, c) for c in f.choices]
            if val is not Select.NULL and val not in f.choices:
                opts.append((val, val))
            w.set_options(opts)
            w.value = val
        elif f.kind == "range":
            lo, hi = ("", "")
            if isinstance(value, (list, tuple)) and len(value) == 2:
                lo, hi = str(value[0]), str(value[1])
            w.value = lo
            self.query_one(f"{wid}-end").value = hi
        else:
            w.value = _to_widget_text(value)

    # -- actions ------------------------------------------------------------------
    def action_save(self) -> None:
        """Save (ctrl+s, the Save button). Asks first when the target is a file
        on disk that this session did not open or write."""
        self._save(lambda path: None)

    def _save(self, then: Callable[[Path], None]) -> None:
        """Write the settings, then run ``then(path)``: at once, or after the
        user agrees to replace a file this session did not open. Invalid
        settings are reported and nothing runs."""
        try:
            if seed_user_defaults(self._slurm_form()):
                self.notify(f"Your Slurm account is now the default for every corpus ({_state_dir() / 'slurm.yaml'})")
            cfg = self._collect()
            self.config = cfg
            path = self._require_project_path()
        except ConfigError as exc:
            self.notify(str(exc), severity="error")
            return
        if path.exists() and path.resolve() not in self._known_settings:
            def answered(go: bool) -> None:
                if go:
                    self._write_settings(cfg, path)
                    then(path)
                else:
                    self.notify("Not saved", severity="warning")
            self.push_screen(ConfirmReplaceSettings(path), answered)
            return
        self._write_settings(cfg, path)
        then(path)

    def _write_settings(self, cfg: Dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f"# lexichron settings file (written by lexichron ui {__version__})\n")
            yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=False)
        self._known_settings.add(path.resolve())
        self.explicit_project = path
        self.loaded_dir = self._corpus_dir(cfg)
        self._saved_slurm = dict(cfg.get("slurm") or {})
        remember_settings_path(path)
        self.notify(f"Saved {path}")
        if getattr(self, "_submit_ready", False):
            self._refresh_sources_line()

    @on(Button.Pressed, "#save")
    def _save_pressed(self) -> None:
        self.action_save()

    @on(Button.Pressed, "#run")
    def _run_pressed(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.notify("A run is already in progress", severity="warning")
            return
        self._save(lambda path: self._guard_overwrite("run", lambda confirmed: self._start_run(path, confirmed)))

    def _start_run(self, project: Path, assume_yes: bool) -> None:
        self.query_one(TabbedContent).active = "tab-run"
        self._run_stage(self.query_one("#runlog", RichLog), project, assume_yes)

    def _overwrite_risk(self) -> Optional[OverwriteRisk]:
        """What the stage, with the settings as saved, would delete first (see
        lexichron.safeguards); None when nothing on disk is at stake. Settings
        the stage cannot be called with are the CLI's to report."""
        try:
            kwargs = build_call(self.func, {k: v for k, v in self.config.items() if k in ("corpus", self.stage)},
                                self.stage)
            return overwrite_risk(self.stage, kwargs)
        except Exception:
            return None

    def _guard_overwrite(self, action: str, then: Callable[[bool], None]) -> None:
        """Run ``then(confirmed)`` once it is safe to: at once, with False, when
        the settings delete nothing; after the user has confirmed in a dialog,
        with True, when they do (the stage is then run with ``--yes``). Keeping
        the data cancels the action. A job (``action == "job"``) is also held
        for a second question when its memory looks too small for its workers."""
        if action == "job":
            inner = then

            def then(confirmed: bool) -> None:
                self._ask_memory(lambda: inner(confirmed))
        self._size_up_overwrite(action, then)

    def _refresh_command_line(self, dest: Optional[Path]) -> None:
        """What a batch script written now would run the stage with: the
        project's own launcher, the image this UI runs in, or the `lexichron`
        on the job's PATH (a development environment, not the image)."""
        cmd = stage_command(dest or Path(SETTINGS_FILENAME), self.stage)
        if cmd[0] == "apptainer":
            text = f"Jobs run in the image {cmd[2]}"
        elif cmd[0] == "lexichron":
            text = ("Jobs run the 'lexichron' on the job's PATH (this environment, not the image); "
                    "for the image, start the UI with lexichron-ui.")
        else:
            text = f"Jobs run with {cmd[0]}"
        self.query_one("#submit-command", Static).update(text)

    def _memory_estimate(self) -> Optional[MemoryEstimate]:
        try:
            kwargs = build_call(self.func, {k: v for k, v in self.config.items() if k in ("corpus", self.stage)},
                                self.stage)
            return memory_estimate(self.stage, kwargs, self._slurm_resolved())
        except Exception:
            return None

    def _ask_memory(self, then: Callable[[], None]) -> None:
        est = self._memory_estimate()
        if est is None or est.fits:
            then()
            return
        self.push_screen(ConfirmMemory(est), lambda go: then() if go else None)

    def _refresh_memory_line(self) -> None:
        line = self.query_one("#submit-memory", Static)
        est = self._memory_estimate()
        line.update(est.describe() if est else "")
        line.set_class(bool(est and not est.fits), "over")

    @work(thread=True, exclusive=True, group="guard")
    def _size_up_overwrite(self, action: str, then: Callable[[bool], None]) -> None:
        risk = self._overwrite_risk()               # walks the database directory: off the UI thread
        self.call_from_thread(self._ask_overwrite, risk, action, then)

    def _ask_overwrite(self, risk: Optional[OverwriteRisk], action: str, then: Callable[[bool], None]) -> None:
        if risk is None:
            then(False)
            return

        def answered(confirmed: Optional[bool]) -> None:
            if confirmed:
                then(True)
            else:
                self.notify(f"Kept {risk.path.name}; nothing run", severity="warning")

        self.push_screen(ConfirmOverwrite(risk, action), answered)

    @work(thread=True, exclusive=True, group="run")
    def _run_stage(self, log: RichLog, project: Path, assume_yes: bool = False) -> None:
        cmd = [sys.executable, "-m", "lexichron.cli", self.stage, str(project)] + (["--yes"] if assume_yes else [])
        self.call_from_thread(log.write, "$ " + " ".join(cmd))
        env = dict(os.environ, PYTHONUNBUFFERED="1", TQDM_MININTERVAL="2")
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, bufsize=1, env=env)
        assert self.proc.stdout is not None
        buf = ""
        while True:
            ch = self.proc.stdout.read(1)
            if not ch:
                break
            if ch in "\r\n":
                if buf.strip():
                    self.call_from_thread(log.write, buf.rstrip())
                buf = ""
            else:
                buf += ch
        rc = self.proc.wait()
        if buf.strip():
            self.call_from_thread(log.write, buf.rstrip())
        self.call_from_thread(log.write, f"[exit {rc}]")
        self.call_from_thread(self.notify, f"Run finished with exit code {rc}",
                              severity="information" if rc == 0 else "error")
        self.call_from_thread(self.refresh_jobs)

    @on(Button.Pressed, "#stop")
    def _stop_pressed(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self.notify("Sent SIGTERM to the running stage", severity="warning")

    def _slurm_cfg(self) -> Dict[str, Any]:
        """What a job of this stage gets, with the Submit tab's values as typed."""
        return resolve_slurm(self.stage, self._collect().get("slurm"))[0]

    def _populate_slurm(self) -> None:
        """Show the stage's resolved Slurm settings on the Submit tab."""
        values = self._slurm_resolved()
        for f in SLURM_FIELDS.fields:
            w = self.query_one(f"#f-slurm-{f.name}", Input)
            text = _to_widget_text(values.get(f.name))
            if w.value != text:
                w.value = text
        self._refresh_sources_line()

    def _stash_slurm(self) -> None:
        """Keep the Submit tab's values for the stage being left, so coming
        back shows them again (they are saved with the next Save)."""
        try:
            layered = stage_override(self.config.get("slurm"), self.stage, self._slurm_form())
        except ConfigError:
            return
        if layered:
            self.config["slurm"] = layered
        else:
            self.config.pop("slurm", None)

    def _refresh_sources_line(self) -> None:
        """Where each Slurm value comes from, as saved; a value typed since
        is marked as such (it becomes the stage's own on Save)."""
        try:
            form = self._slurm_form()
        except ConfigError as exc:
            self.query_one("#submit-sources", Static).update(str(exc))
            return
        values, sources = resolve_slurm(self.stage, self._saved_slurm)
        for key, typed in form.items():
            if key == "jobs" and self.stage not in ARRAY_STAGES:
                continue
            if str(typed) != str(values.get(key, "")):
                values[key] = typed
                sources[key] = f"typed here; saved for {self.stage} on Save"
        for key in [k for k in values if k not in form and k in STAGE_KEYS + USER_KEYS]:
            if key == "jobs" and self.stage not in ARRAY_STAGES:
                continue
            values.pop(key)                        # cleared on the form
            sources.pop(key, None)
        self.query_one("#submit-sources", Static).update(describe_sources(values, sources))

    @on(Input.Changed, "#f-slurm-account, #f-slurm-partition, #f-slurm-cpus, #f-slurm-mem, #f-slurm-time, "
                       "#f-slurm-jobs, #f-slurm-job_name")
    def _slurm_field_changed(self) -> None:
        if getattr(self, "_submit_ready", False):
            self._refresh_sources_line()

    @on(Button.Pressed, "#slurm-defaults")
    def _slurm_defaults_pressed(self) -> None:
        """Make the account and partition on the form the user's defaults for
        every corpus; this corpus's file then no longer needs its own copy."""
        try:
            form = self._slurm_form()
        except ConfigError as exc:
            self.notify(str(exc), severity="error")
            return
        if not form.get("account"):
            self.notify("Fill in the account first", severity="warning")
            return
        path = write_user_defaults(form)
        for key in USER_KEYS:
            self._saved_slurm.pop(key, None)
        section = dict(self.config.get("slurm") or {})
        for key in USER_KEYS:
            section.pop(key, None)
        if section:
            self.config["slurm"] = section
        else:
            self.config.pop("slurm", None)
        self._refresh_sources_line()
        self.notify(f"Saved your Slurm defaults to {path}")

    @on(Button.Pressed, "#write-sbatch")
    def _write_sbatch_pressed(self) -> None:
        self._save(lambda path: self._guard_overwrite("job", self._write_sbatch))

    def _clear_for_array(self, assume_yes: bool) -> Optional[str]:
        """A restart submitted as a job array cannot delete in the job (every
        task would delete what its siblings are writing), so once the user
        has confirmed the deletion it is done here, now, and the tasks
        resume into the empty directory. Returns a line saying so, or None
        when nothing was deleted."""
        if not assume_yes or array_size(self.stage, self._slurm_cfg()) <= 1:
            return None
        try:
            kwargs = build_call(self.func, {k: v for k, v in self.config.items() if k in ("corpus", self.stage)},
                                self.stage)
        except Exception:
            return None
        target = overwrite_target(self.stage, kwargs)
        if target is None or not target.exists():
            return None
        shutil.rmtree(target)
        return f"Deleted {target} now; the array's {array_size(self.stage, self._slurm_cfg())} tasks resume into it."

    def _write_sbatch(self, assume_yes: bool) -> None:
        try:
            cleared = self._clear_for_array(assume_yes)
            path = write_sbatch(self.project_path, self.stage, self._slurm_cfg(), assume_yes=assume_yes)
        except (ConfigError, OSError) as exc:
            self.query_one("#submit-status", Static).update(f"Could not write script: {exc}")
            return
        self.query_one("#submit-status", Static).update(
            (f"{cleared}\n" if cleared else "") + f"Wrote {path}\nSubmit by hand with:  sbatch {path}")

    @on(Button.Pressed, "#submit")
    def _submit_pressed(self) -> None:
        self._save(lambda path: self._guard_overwrite("job", self._submit))

    def _submit(self, assume_yes: bool) -> None:
        status = self.query_one("#submit-status", Static)
        try:
            if not self.bridge:
                raise BridgeUnavailable("no way to run sbatch here (LEXICHRON_BRIDGE_DIR not set and sbatch not on PATH)")
            cleared = self._clear_for_array(assume_yes)
            path = write_sbatch(self.project_path, self.stage, self._slurm_cfg(), assume_yes=assume_yes)
            job_id = self.bridge.sbatch(path)
        except (ConfigError, OSError, BridgeUnavailable, RuntimeError, TimeoutError) as exc:
            status.update(f"Submit failed: {exc}")
            self.notify("Submit failed", severity="error")
            return
        tasks = array_size(self.stage, self._slurm_cfg())
        what = f"job array {job_id} ({tasks} tasks)" if tasks > 1 else f"job {job_id}"
        status.update((f"{cleared}\n" if cleared else "") + f"Submitted {what}  ({path})")
        self.notify(f"Submitted {what}")
        self.query_one(TabbedContent).active = "tab-jobs"
        self.refresh_jobs()

    @on(Button.Pressed, "#refresh-jobs")
    def _refresh_pressed(self) -> None:
        self.refresh_jobs()

    @on(Button.Pressed, "#cancel-job")
    def _cancel_pressed(self) -> None:
        table = self.query_one("#jobs", DataTable)
        if table.cursor_row is None or table.row_count == 0 or not self.bridge:
            return
        job_id = str(table.get_row_at(table.cursor_row)[0])
        try:
            self.bridge.scancel(job_id)
            self.notify(f"Cancelled {job_id}")
        except (BridgeUnavailable, TimeoutError) as exc:
            self.notify(str(exc), severity="error")
        self.refresh_jobs()

    @on(Button.Pressed, "#clear-runs")
    def _clear_runs_pressed(self) -> None:
        """Remove the run directories of runs that are no longer running (done,
        failed, or stopped); a running run's progress file is in use."""
        import shutil
        removed = 0
        for d in self._run_dirs():
            p = read_progress(d / "progress.json")
            if p and self._still_running(p):
                continue
            try:
                shutil.rmtree(d)
                removed += 1
            except OSError as exc:
                self.notify(f"Could not remove {d.name}: {exc}", severity="error")
        self.selected_run = None
        self.refresh_jobs()
        self.notify(f"Removed {removed} run{'s' if removed != 1 else ''}")

    def _still_running(self, doc: Dict[str, Any]) -> bool:
        """Whether a run may still be going: its record says running (whatever
        the phase label says: a long phase such as training shows its own
        name) and nothing says otherwise. Slurm saying the job has left the
        queue settles it. A run with no job is a local one: ours if its process
        is alive; otherwise it is trusted while its progress file is fresh
        and taken as dead once that goes stale."""
        if doc.get("state", "running") != "running":
            return False
        info = self._summarize(doc)
        if info["state_label"] == "stopped":
            return False
        if doc.get("slurm_job_id"):
            return True
        if self.proc is not None and self.proc.poll() is None:
            return True
        return not info["stale"]

    # -- jobs & runs ----------------------------------------------------------------
    SQUEUE_EVERY_S = 15.0        # Slurm is asked this often; the run files every refresh

    def refresh_jobs(self) -> None:
        """Periodic refresh. All the waiting (Slurm through the bridge, the
        run directories and progress files on a networked filesystem, the
        log tail) happens in a worker thread; the UI thread only paints the
        snapshot, so typing never stalls behind a slow squeue. A refresh
        that is still waiting is left alone rather than joined by another:
        a slow squeue must not be asked again before it has answered."""
        if self._gathering:
            return
        self._gathering = True
        self._gather_status()

    @work(thread=True, group="status")
    def _gather_status(self) -> None:
        try:
            snap = self._collect_status()
        except Exception as exc:                      # pragma: no cover - defensive
            logging.getLogger(__name__).warning("refresh failed: %s", exc)
            snap = {"error": str(exc)}
        finally:
            self._gathering = False
        self.call_from_thread(self._apply_status, snap)

    def _collect_status(self) -> Dict[str, Any]:
        """Everything the Progress tab shows, gathered off the UI thread."""
        import time as _time
        jobs: Optional[List[dict]] = self._status.get("jobs")
        jobs_error = self._status.get("jobs_error", "")
        if self.bridge and self.bridge.available() and \
                _time.monotonic() - self._squeue_at >= self.SQUEUE_EVERY_S:
            try:
                jobs = self.bridge.squeue(user=getpass.getuser())
                jobs_error = ""
            except (BridgeUnavailable, TimeoutError) as exc:
                jobs_error = str(exc)
            self._squeue_at = _time.monotonic()
        dirs = self._run_dirs()
        docs = [(d, read_progress(d / "progress.json")) for d in dirs]
        selected = self.selected_run if self.selected_run in dirs else (dirs[0] if dirs else None)
        sel_doc = next((doc for d, doc in docs if d == selected), None)
        log_lines = tail_lines(sel_doc.get("log_path"), 15) if sel_doc else []
        return {"jobs": jobs, "jobs_error": jobs_error, "docs": docs, "selected": selected,
                "log_lines": log_lines}

    def _apply_status(self, snap: Dict[str, Any]) -> None:
        if "error" in snap:
            try:
                self.query_one("#run-title", Static).update(f"(refresh failed: {snap['error']}; retrying)")
            except Exception:
                pass
            return
        self._status = snap
        self._follow_running_ensemble(snap["docs"])
        table = self.query_one("#jobs", DataTable)
        table.clear()
        # Job ids of queued/running jobs, or None when Slurm cannot be asked.
        self.queued_jobs = None
        if snap["jobs"] is not None:
            ours = {str(doc["slurm_job_id"]) for _, doc in snap["docs"] if doc and doc.get("slurm_job_id")}
            rows = sorted(snap["jobs"], key=lambda r: 0 if r["job_id"] in ours else 1)   # lexichron jobs first
            self.queued_jobs = {r["job_id"] for r in rows}
            for row in rows:
                table.add_row(row["job_id"], row["name"], row["state"], row["elapsed"],
                              row["limit"], row["nodes"], row["reason"])
        elif snap["jobs_error"]:
            table.add_row("-", snap["jobs_error"][:40], "", "", "", "", "")
        # The snapshot was gathered for the run selected when the gathering
        # began. A run the user selected since wins: keep it, paint what the
        # snapshot knows about it, and fetch its log tail with the next refresh.
        dirs = [d for d, _ in snap["docs"]]
        if self.selected_run not in dirs:
            self.selected_run = snap["selected"]
        stale = snap["selected"] != self.selected_run
        if stale:
            snap = dict(snap, selected=self.selected_run, log_lines=None)
            self._status = snap
        self._refresh_runs(snap["docs"])
        self._refresh_run_panel(snap)
        if stale:
            self.refresh_jobs()

    def _summarize(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """summarize() plus what Slurm knows: a 'running' run whose job has
        left the queue is reported as stopped."""
        job = doc.get("slurm_job_id")
        gone = (self.queued_jobs is not None and job is not None
                and str(job) not in self.queued_jobs)
        return summarize(doc, job_gone=gone)

    def _run_dirs(self) -> List[Path]:
        """Run directories with a progress file, newest first. Networked file
        systems can answer a stat with ESTALE while a run is replacing its
        progress.json, so every probe treats OSError as 'not there right now'."""
        proj = self.project_path
        run_root = proj.parent / ".lexichron" / "runs" if proj else None
        try:
            if not run_root or not run_root.is_dir():
                return []
            dirs = []
            for d in run_root.iterdir():
                try:
                    if (d / "progress.json").exists():
                        dirs.append(d)
                except OSError:
                    dirs.append(d)       # being rewritten; read_progress will cope
            return sorted(dirs, reverse=True)[:20]
        except OSError:
            return []

    def _refresh_runs(self, docs: List[tuple]) -> None:
        """Paint the Runs table. While the set of runs is unchanged the cells
        are updated in place: rebuilding the rows would move the cursor, and
        a cursor move is indistinguishable from the user selecting a run."""
        runs = self.query_one("#runs", DataTable)
        rows = []
        for d, p in docs:
            if not p:
                continue
            info = self._summarize(p)
            files = f"{info['done']}/{info['total']}" if info["total"] else str(info["done"])
            if info["failed"]:
                files += f" ({info['failed']} failed)"
            # Records: what the compacted database holds once the run is done
            # (marked *), the records kept so far while it runs
            db = p.get("database") or {}
            if p.get("stage") in ("tune", "train"):
                # models, not records: the rate is models per hour and there is no database count
                records = "–"
                rate = f"{info['done'] / info['elapsed_s'] * 3600:.1f}/h" if info["elapsed_s"] else "–"
            else:
                if info["state"] == "done" and db.get("records") is not None:
                    records = f"{int(db['records']):,}" + ("*" if db.get("compacted") else " (est.)")
                else:
                    records = f"{int(p.get('entries_written', 0)):,}"
                rate = f"{fmt_count(info['rate'])}/s"
            rows.append((str(d), (d.name, info["state_label"], files, records, rate,
                                  fmt_duration(info["elapsed_s"]), p.get("slurm_job_id") or "-")))
        keys = [k for k, _ in rows]
        if [k.value for k in runs.rows] == keys:
            for key, values in rows:
                for col, value in zip(self._run_columns, values):
                    runs.update_cell(key, col, value, update_width=True)
            return
        runs.clear()
        for key, values in rows:
            runs.add_row(*values, key=key)
        if self.selected_run and str(self.selected_run) in keys:
            runs.move_cursor(row=keys.index(str(self.selected_run)), animate=False)

    @on(DataTable.RowHighlighted, "#runs")
    def _run_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key is not None and event.row_key.value:
            chosen = Path(event.row_key.value)
            if chosen == self.selected_run:
                return
            self.selected_run = chosen
            # paint what is already known; the log tail arrives with the next refresh
            snap = dict(self._status, selected=chosen, log_lines=None)
            self._refresh_run_panel(snap)
            self.refresh_jobs()

    def _refresh_run_panel(self, snap: Dict[str, Any]) -> None:
        title = self.query_one("#run-title", Static)
        bar = self.query_one("#run-bar", ProgressBar)
        log = self.query_one("#run-log", RichLog)
        if not self.selected_run:
            if self.project_path is None:
                title.update("Fill in the corpus directory, language and n-gram size on the Settings tab; "
                             "that corpus's runs will appear here.")
            else:
                title.update("No runs yet. Use 'Run here' or 'Submit to Slurm'; runs appear here as they start.")
            for wid in ("#run-headline", "#run-inputs", "#run-message"):
                self.query_one(wid, Static).update("")
            bar.update(total=None, progress=0)
            return
        doc = next((d for p, d in snap.get("docs", []) if p == self.selected_run), None)
        if not doc:
            title.update(f"{self.selected_run.name}: progress file unreadable")
            return
        info = self._summarize(doc)
        title.update(f"Run {self.selected_run.name}" + (f"  ->  {info['db_path']}" if info["db_path"] else ""))
        if info["total"]:
            bar.update(total=info["total"], progress=info["done"] + info["failed"])
        else:
            bar.update(total=None, progress=0)
        self.query_one("#run-headline", Static).update(info["headline"])
        self.query_one("#run-inputs", Static).update(info["inputs_text"] or " ")
        self.query_one("#run-message", Static).update(info["message"])
        self.query_one("#meters", Meters).show(meters(doc, info))
        # Log tail: only rewrite when it changes, so the pane does not flicker.
        lines = snap.get("log_lines")
        if lines is None:                       # not fetched yet for this run
            return
        key = (self.selected_run, tuple(lines))
        if key != self._log_shown:
            log.clear()
            if lines:
                for line in lines:
                    log.write(line)
            else:
                log.write("(no log yet)" if info["log_path"] is None else f"(cannot read {info['log_path']})")
            self._log_shown = key


    # -- Inspect tab --------------------------------------------------------
    def _refresh_inspect_candidates(self) -> None:
        """List the corpus directory's databases and vocabulary files in the
        Database selector; the path field accepts anything else."""
        corpus_dir = self._corpus_dir(self.config)
        if corpus_dir == self._inspect_dir:
            return
        self._inspect_dir = corpus_dir
        sel = self.query_one("#inspect-db", Select)
        sel.set_options(candidates(str(corpus_dir) if corpus_dir else None))

    # -- Inspect tab, tune stage: a search's scores --------------------------
    def _refresh_searches(self) -> None:
        """List the corpus's searches, newest first; pick the one the settings
        name, else the newest, and show it."""
        corpus_dir = self._corpus_dir(self.config)
        found = searches(str(corpus_dir) if corpus_dir else None)
        sel = self.query_one("#search-name", Select)
        current = sel.value if sel.value is not Select.NULL else None
        sel.set_options(found)
        wanted = (self.config.get("tune") or {}).get("name")
        choice = next((path for name, path in found if name == wanted), None) or \
            (current if any(path == current for _, path in found) else None) or (found[0][1] if found else None)
        if choice is None:
            self._show_search(None)
            return
        if sel.value != choice:
            sel.value = choice                           # Select.Changed loads it
        else:
            self._load_search(choice)

    @on(Select.Changed, "#search-name")
    def _search_chosen(self, event: Select.Changed) -> None:
        if event.value is not Select.NULL:
            self._load_search(str(event.value))

    @on(Button.Pressed, "#search-reload")
    def _search_reload_pressed(self) -> None:
        self._refresh_searches()

    @on(Select.Changed, "#search-view")
    @on(Switch.Changed, "#search-spreads")
    def _search_columns_changed(self) -> None:
        if self._search is not None:
            self._show_search(self._search)

    @work(thread=True, group="search", exclusive=True)
    def _load_search(self, path: str) -> None:
        search = read_search(path)
        self.call_from_thread(self._show_search, search)

    def _show_search(self, search: Optional[Dict[str, Any]]) -> None:
        self._search = search
        table = self.query_one("#search-table", DataTable)
        table.clear(columns=True)
        self._show_search_cell(None)
        empty = self.query_one("#search-empty", Static)
        has = search is not None and not search.get("error")
        for wid in ("#search-meters", "#search-grid", "#search-table", "#search-legend"):
            self.query_one(wid).display = has
        if not has:
            empty.update(search["error"] if search else
                         "No tuning search in this corpus directory yet: run the tune stage, and its scores "
                         "appear here as they are produced.")
            empty.display = True
            return
        empty.display = False
        overview = search_overview(search)
        view = self.query_one("#search-view", Select).value
        view = str(view) if view is not Select.NULL else "all"
        starred, rule = suggested_for(search, view)
        # the tile follows the family selector: that family's pick alone, under
        # its rule's name; on "all", the overall pick and the rule it came from
        tiles = list(overview["tiles"])
        picks = search.get("suggestions") or {}
        overall_family = "axis" if "axis" in picks else "neighbourhoods" if "neighbourhoods" in picks else None
        family = view if view != "all" else overall_family
        tiles[3] = ("Suggested cell", starred or "–",
                    f"by the {FAMILY_LABELS[family]} rule" if family and starred else "")
        self.query_one("#search-meters", Meters).show([("", tiles)])
        self.query_one("#search-grid", Static).update(overview["grid"])
        self.query_one("#search-legend-text", Static).update(
            STABILITY_NOTE + " Click a heading to sort by it (again to reverse); select a cell for its years."
            + (f"\n★ {starred}: {rule}." if starred and rule else ""))
        headings, grid, stems = search_table(search, view=view,
                                             spreads=self.query_one("#search-spreads", Switch).value)
        best = best_in_columns(search, headings)
        if not (set(headings) - {"", "Years"} - {h for k, h, _ in SEARCH_COLUMNS if k in PARAM_KEYS}):
            self.query_one("#search-grid", Static).update(
                overview["grid"] + "\n(no scores of this kind yet: a run on this version adds them to every "
                "cell whose models are on disk)")
        for h in headings:
            table.add_column(h, key=h or "best")
        # numbers are right-aligned by padding, not Text(justify=...): the
        # DataTable re-renders a hovered row by another path, which drops the
        # justification and clips the cell (a column's width is its widest value)
        widths = [max([len(h)] + [len(r[i]) for r in grid]) for i, h in enumerate(headings)]
        for row, stem in zip(grid, stems):
            cells = [Text(row[0], style="bold yellow")]
            for i, (h, v) in enumerate(zip(headings[1:], row[1:]), start=1):
                style = "bold" if best.get(h) == v.split(" ")[0] and v != "–" else ""      # "0.600 ±.012"
                cells.append(Text(_aligned(v, widths[i]), style=style))
            table.add_row(*cells, key=stem)
        col, reverse = self._search_sort
        if col is not None and col in headings:
            self._sort_search_table(col, reverse)
        if stems:                                        # the view's suggested cell's years to begin with
            chosen = starred if starred in stems else stems[0]
            self._show_search_cell(chosen)
            try:
                table.move_cursor(row=table.get_row_index(chosen))
            except Exception:
                pass

    def _show_search_cell(self, stem: Optional[str]) -> None:
        """The selected cell's per-year rows and year pairs under the table."""
        years = self.query_one("#search-years", DataTable)
        years.clear(columns=True)
        title, pairs = self.query_one("#search-cell", Static), self.query_one("#search-pairs", Static)
        if not stem or not self._search:
            title.display = years.display = pairs.display = False
            return
        headings, rows = cell_years(self._search, stem)
        title.update(f"{stem} by year")
        title.display = True
        years.display = bool(rows)
        if rows:
            for h in headings:
                years.add_column(h)
            widths = [max([len(h)] + [len(r[i]) for r in rows]) for i, h in enumerate(headings)]
            for row in rows:
                years.add_row(*(Text(_aligned(v, w)) for v, w in zip(row, widths)))
        text = cell_pairs(self._search, stem)
        pairs.update(text or ("No rows for this cell yet" if not rows else ""))
        pairs.display = bool(text) or not rows

    @on(DataTable.HeaderSelected, "#search-table")
    def _search_header_selected(self, event: DataTable.HeaderSelected) -> None:
        """Sort by the clicked column; clicking it again reverses the order."""
        col = str(event.column_key.value)
        current, reverse = self._search_sort
        reverse = not reverse if current == col else col not in ("Dims", "Window", "Min count", "Epochs",
                                                                  "Approach", "Weighting", "best")
        self._search_sort = (col, reverse)
        self._sort_search_table(col, reverse)

    def _sort_search_table(self, col: str, reverse: bool) -> None:
        def key(value):
            text = str(value).replace(",", "").strip().split(" ")[0]        # "0.590 ±.012" sorts by 0.590
            try:
                return (1, float(text))
            except ValueError:
                return (0, text)                          # "–" and labels sort before numbers
        self.query_one("#search-table", DataTable).sort(col, key=key, reverse=reverse)

    @on(DataTable.RowSelected, "#search-table")
    def _search_row_selected(self, event: DataTable.RowSelected) -> None:
        if self._search and event.row_key is not None and event.row_key.value:
            self._show_search_cell(str(event.row_key.value))

    @on(Select.Changed, "#inspect-db")
    def _inspect_db_chosen(self, event: Select.Changed) -> None:
        if event.value is not Select.NULL:
            self.query_one("#inspect-path", Input).value = str(event.value)
            self._inspect("describe")

    @on(Input.Submitted, "#inspect-path")
    def _inspect_path_submitted(self) -> None:
        self._inspect("describe")

    @on(Input.Submitted, "#inspect-key")
    def _inspect_key_submitted(self) -> None:
        self._inspect("key")

    @on(Button.Pressed, "#inspect-head")
    def _inspect_head_pressed(self) -> None:
        self._inspect("head")

    @on(Button.Pressed, "#inspect-from-key")
    def _inspect_key_pressed(self) -> None:
        self._inspect("key")

    @on(Button.Pressed, "#inspect-prefix")
    def _inspect_prefix_pressed(self) -> None:
        self._inspect("prefix")

    def _inspect(self, mode: str) -> None:
        path = self.query_one("#inspect-path", Input).value.strip()
        if not path:
            self.query_one("#inspect-summary", Static).update("Choose a database or type a path.")
            return
        key = self.query_one("#inspect-key", Input).value
        try:
            n = max(1, int(self.query_one("#inspect-rows", Input).value or 20))
        except ValueError:
            n = 20
        layout = str(self.query_one("#inspect-layout", Select).value or "auto")
        self.query_one("#inspect-note", Static).update("Reading...")
        self._inspect_worker(path, mode, key, n, layout)

    @work(thread=True, group="inspect", exclusive=True)
    def _inspect_worker(self, path: str, mode: str, key: str, n: int, layout: str) -> None:
        """Open and read off the UI thread: a networked filesystem can take
        seconds to open a database for the first time."""
        info = describe(path)
        found: List[tuple] = []
        note = ""
        used = info.get("layout", layout)
        if not info.get("error") and mode != "describe":
            try:
                used, found = inspect_rows(path, mode, key, n, layout, raw=True)
                note = (f"{len(found)} record(s)" + (f" from {key!r}" if mode == "key" else
                        f" with prefix {key!r}" if mode == "prefix" else "") + f", keys read as {used}")
                if not found:
                    note = "No records" + (f" from {key!r}" if mode == "key" else
                                           f" with prefix {key!r}" if mode == "prefix" else "")
            except Exception as exc:
                note = str(exc)
        self.call_from_thread(self._show_inspect, info, used, found, note)

    def _show_inspect(self, info: Dict[str, Any], layout: str, found: List[tuple], note: str) -> None:
        self.query_one("#inspect-summary", Static).update(format_summary(info))
        prov = info.get("provenance")
        box = self.query_one("#inspect-provenance-box", Collapsible)
        box.display = bool(prov)
        if prov:
            self.query_one("#inspect-provenance", Static).update("\n".join(format_provenance(prov)))
        self.query_one("#inspect-note", Static).update(note)
        self._inspect_found = [(layout, rk, rv) for _, _, rk, rv in found]
        table = self.query_one("#inspect-table", DataTable)
        table.clear(columns=True)
        if layout == "table" and found:
            cols = found[0][0].split("\t")                   # the header travels as the first row
            table.add_columns(*cols)
            for k, v, _rk, _rv in found[1:]:
                cells = [k] + v.split("\t")
                table.add_row(*(cells + [""] * (len(cols) - len(cells)))[:len(cols)])
        else:
            table.add_columns("Key", "Value")
            for k, v, _rk, _rv in found:
                table.add_row(k, v)
        self.query_one("#inspect-detail", Static).update("")

    @on(DataTable.RowSelected, "#inspect-table")
    def _inspect_row_selected(self, event: DataTable.RowSelected) -> None:
        """Enter or click on an n-gram-keyed row: its full year series."""
        rows = getattr(self, "_inspect_found", [])
        if event.cursor_row is None or event.cursor_row >= len(rows):
            return
        layout, _rk, rv = rows[event.cursor_row]
        if layout != "ngram":
            return
        series = year_series(rv)
        text = "  ".join(f"{y}: {c:,}/{v:,}" for y, c, v in series)
        self.query_one("#inspect-detail", Static).update(f"Year series (count/volumes), {len(series)} years:\n{text}")


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="lexichron ui", description="Terminal UI for lexichron.")
    parser.add_argument("project", nargs="?", default=None,
                        help=f"settings YAML file to open (default: the {SETTINGS_FILENAME} beside the database)")
    parser.add_argument("--stage", default="acquire", choices=sorted(STAGES))
    args = parser.parse_args(argv)
    LexichronApp(args.project, args.stage).run()
    return 0
