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
  Submit   — Slurm resources; writes the batch script and submits it through
             the host bridge when one is running
  Progress — the latest runs (from each run's progress.json) with a progress
             bar, rate, time left and the tail of the run's log; plus squeue
             for your Slurm jobs
"""
from __future__ import annotations

import getpass
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button, Collapsible, DataTable, Footer, Header, Input, Label, ProgressBar,
    RichLog, Select, Static, Switch, TabbedContent, TabPane,
)

from lexichron import __version__
from lexichron.bridge import BridgeUnavailable, HostBridge, default_bridge_dir
from lexichron.cli import STAGES, _format_call, _resolve
from lexichron.config import ConfigError, build_call, load_project
from lexichron.ui.progress_view import fmt_count, fmt_duration, summarize, tail_lines
from lexichron.ui.schema import Field, SLURM_FIELDS, Section, stage_sections
from lexichron.ui.slurm import write_sbatch
from ngramprep.ngram_acquire.db.build_path import build_db_path
from ngramprep.ngram_acquire.progress import read_progress

__all__ = ["LexichronApp", "main", "SETTINGS_FILENAME"]

SETTINGS_FILENAME = "lexichron.yaml"


def _state_dir() -> Path:
    """Per-user state (the last settings file opened); ~/.lexichron by default."""
    return Path(os.environ.get("LEXICHRON_STATE_DIR") or Path.home() / ".lexichron")


def remember_settings_path(path: Path) -> None:
    try:
        d = _state_dir()
        d.mkdir(parents=True, exist_ok=True)
        (d / "last_settings").write_text(str(path) + "\n", encoding="utf-8")
    except OSError:
        pass


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
    if field.kind == "int" or (field.kind == "choice" and field.name == "ngram_size"):
        return int(str(raw).strip())
    if field.kind == "float":
        return float(str(raw).strip())
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


class FixedValue(Static):
    """A setting with exactly one valid value: shown as text, still saved."""

    def __init__(self, value: str, **kwargs) -> None:
        super().__init__(value, **kwargs)
        self.value = value


# ---------------------------------------------------------------------------
# The app
# ---------------------------------------------------------------------------

class LexichronApp(App):
    TITLE = f"lexichron {__version__}"
    CSS = """
    #form { height: 1fr; padding: 0 1; scrollbar-size-vertical: 1; }
    .section { text-style: bold; color: $accent; margin-top: 1; }
    .field-help { color: $text-muted; padding-left: 22; margin-bottom: 1; display: none; }
    .show-help .field-help { display: block; }
    .row { height: auto; }
    .row Label { width: 22; }
    .row Input, .row Select { width: 1fr; }
    .row .range-sep { width: auto; padding: 0 1; color: $text-muted; }
    .row Switch { height: 1; border: none; padding: 0; }
    .row .spacer { width: 1fr; }
    .hint { height: 3; content-align: left middle; color: $text-muted; }
    .row FixedValue { width: 1fr; color: $text-muted; }
    .row .help-mark { width: 3; color: $text-muted; text-align: center; }
    Collapsible { margin-top: 1; }
    CollapsibleTitle { padding: 0; }
    #dest { padding: 0 1; height: auto; }
    #call { padding: 0 1; }
    #status { color: $warning; padding: 0 1; height: auto; }
    .actions { height: auto; padding: 1 1 0 1; }
    .actions Button { margin-right: 2; }
    #runlog { height: 1fr; }
    #jobs { height: auto; max-height: 8; }
    #runs { height: auto; max-height: 8; }
    #run-panel { height: auto; padding: 0 1; border-top: solid $secondary; }
    #run-title { text-style: bold; }
    #run-bar { width: 1fr; margin: 0 0 1 0; }
    #run-bar Bar { width: 1fr; }
    #run-headline, #run-detail, #run-current, #run-corpus { height: auto; }
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
        self.stage = stage
        self.func = _resolve(STAGES[stage][0])
        self.sections: List[Section] = stage_sections(self.func, stage) + [SLURM_FIELDS]
        self.config: Dict[str, Any] = {}
        if self.explicit_project and self.explicit_project.exists():
            self.config = load_project(self.explicit_project)
        self.loaded_dir: Optional[Path] = self._corpus_dir(self.config)
        bridge_dir = default_bridge_dir()
        self.bridge: Optional[HostBridge] = HostBridge(bridge_dir) if bridge_dir else None
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

    def _corpus_dir(self, config: Dict[str, Any]) -> Optional[Path]:
        """The directory of the database these settings describe, or None
        until the corpus directory, release, language and n-gram size are set."""
        corpus = config.get("corpus") or {}
        stage = config.get(self.stage) or {}
        stub, release, lang = corpus.get("db_path_stub"), corpus.get("release"), corpus.get("language")
        n = stage.get("ngram_size")
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
        with TabbedContent(initial="tab-project"):
            with TabPane("Settings", id="tab-project"):
                yield Static("", id="dest")
                yield Static("", id="status")
                with VerticalScroll(id="form"):
                    for sec in self.sections:
                        if sec.name == "slurm":
                            continue
                        yield Static(sec.name.capitalize(), classes="section")
                        for f in sec.fields:
                            if not f.advanced:
                                yield from self._field_widgets(sec.name, f)
                    advanced = [(sec.name, f) for sec in self.sections for f in sec.fields if f.advanced]
                    if advanced:
                        with Collapsible(title="Advanced", collapsed=True, id="advanced"):
                            for sec_name, f in advanced:
                                yield from self._field_widgets(sec_name, f)
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
                    with Horizontal(classes="actions"):
                        yield Button("Write batch script", id="write-sbatch")
                        yield Button("Submit to Slurm", id="submit", variant="primary")
                    yield Static("", id="submit-status")
            with TabPane("Progress", id="tab-jobs"):
                with VerticalScroll():
                    yield Static("Runs (select one to see its progress)", classes="section")
                    yield DataTable(id="runs")
                    with Vertical(id="run-panel"):
                        yield Static("", id="run-title")
                        yield ProgressBar(total=None, show_eta=False, id="run-bar")
                        yield Static("", id="run-headline")
                        yield Static("", id="run-detail", classes="muted")
                        yield Static("", id="run-current", classes="muted")
                        yield Static("", id="run-corpus", classes="muted")
                        yield Static("", id="run-message")
                        yield RichLog(id="run-log", wrap=False, highlight=False, markup=False)
                    yield Static("Slurm jobs", classes="section")
                    yield DataTable(id="jobs")
                    with Horizontal(classes="actions"):
                        yield Button("Refresh", id="refresh-jobs")
                        yield Button("Cancel selected job", id="cancel-job", variant="error")
        yield Footer()

    def _field_widgets(self, section: str, f: Field):
        wid = f"f-{section}-{f.name}"
        current = (self.config.get(section) or {}).get(f.name, f.default)
        label = f.title + (" *" if f.required else "")
        with Horizontal(classes="row"):
            yield Label(label)
            if f.kind == "bool":
                w = Switch(value=bool(current), id=wid)
                w.tooltip = f.help or None
                yield w
                w = Static("", classes="spacer")
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
                hint = {"path": "directory", "list": "comma-separated", "int": "number"}.get(f.kind, f.kind)
                w = Input(value=_to_widget_text(current), id=wid, compact=True,
                          placeholder=f"({hint}{', optional' if not f.required else ''})")
            w.tooltip = f.help or None
            yield w
            mark = Static("?" if f.help else "", classes="help-mark")
            mark.tooltip = f.help or None
            yield mark
        if f.help:
            yield Static(f.help, classes="field-help")

    def action_toggle_help(self) -> None:
        """F1: show or hide the help line under every setting."""
        self.screen.toggle_class("show-help")

    def on_mount(self) -> None:
        jobs = self.query_one("#jobs", DataTable)
        jobs.add_columns("Job", "Name", "State", "Elapsed", "Limit", "Nodes", "Reason")
        jobs.cursor_type = "row"
        runs = self.query_one("#runs", DataTable)
        runs.add_columns("Run", "State", "Files", "Entries", "Rate", "Elapsed", "Job")
        runs.cursor_type = "row"
        self.selected_run: Optional[Path] = None
        self._log_shown: Optional[tuple] = None
        self.queued_jobs: Optional[set] = None
        self._refresh_preview()
        self.refresh_jobs()
        self.set_interval(5, self.refresh_jobs)
        if not self.bridge or not self.bridge.available():
            self.query_one("#submit-status", Static).update(
                "No host helper: 'Submit' and the job table need the UI started with the "
                "lexichron-ui command (not from inside the image). 'Write batch script' still works.")

    # -- form -> config -----------------------------------------------------------
    def _collect(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {}
        for sec in self.sections:
            vals: Dict[str, Any] = {}
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
        dest = self.project_path
        self.query_one("#dest", Static).update(
            f"Settings file: {dest if dest else '(fill in the corpus directory, language and n-gram size)'}"
            + ("  (reopened from last time)" if self.reopened else ""))
        try:
            kwargs = build_call(self.func, {k: v for k, v in cfg.items() if k != "slurm"}, self.stage)
        except ConfigError as exc:
            call.update("(fill in the required settings)")
            status.update(self._friendly(str(exc)))
            return
        call.update(_format_call(self.func, kwargs))
        status.update("")

    def _friendly(self, message: str) -> str:
        """Replace ``section.key`` names in a config error with the form's labels."""
        for sec in self.sections:
            for f in sec.fields:
                message = message.replace(f"{sec.name}.{f.name}", f.title)
        return message

    @on(Input.Changed)
    @on(Select.Changed)
    @on(Switch.Changed)
    def _on_any_change(self, _event) -> None:
        self._refresh_preview()

    # -- actions ------------------------------------------------------------------
    def action_save(self) -> None:
        try:
            cfg = self._collect()
            self.config = cfg
            path = self._require_project_path()
        except ConfigError as exc:
            self.notify(str(exc), severity="error")
            raise
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f"# lexichron settings file (written by lexichron ui {__version__})\n")
            yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=False)
        self.explicit_project = path
        self.loaded_dir = self._corpus_dir(cfg)
        remember_settings_path(path)
        self.notify(f"Saved {path}")

    def _save_quietly(self) -> Optional[Path]:
        try:
            self.action_save()
        except ConfigError:
            return None
        return self.project_path

    @on(Button.Pressed, "#save")
    def _save_pressed(self) -> None:
        self._save_quietly()

    @on(Button.Pressed, "#run")
    def _run_pressed(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.notify("A run is already in progress", severity="warning")
            return
        path = self._save_quietly()
        if path is None:
            return
        self.query_one(TabbedContent).active = "tab-run"
        self._run_stage(self.query_one("#runlog", RichLog), path)

    @work(thread=True, exclusive=True, group="run")
    def _run_stage(self, log: RichLog, project: Path) -> None:
        cmd = [sys.executable, "-m", "lexichron.cli", self.stage, str(project)]
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
        return self._collect().get("slurm", {})

    @on(Button.Pressed, "#write-sbatch")
    def _write_sbatch_pressed(self) -> None:
        try:
            self.action_save()
            path = write_sbatch(self.project_path, self.stage, self._slurm_cfg())
        except (ConfigError, OSError) as exc:
            self.query_one("#submit-status", Static).update(f"Could not write script: {exc}")
            return
        self.query_one("#submit-status", Static).update(
            f"Wrote {path}\nSubmit by hand with:  sbatch {path}")

    @on(Button.Pressed, "#submit")
    def _submit_pressed(self) -> None:
        status = self.query_one("#submit-status", Static)
        try:
            self.action_save()
            path = write_sbatch(self.project_path, self.stage, self._slurm_cfg())
            if not self.bridge:
                raise BridgeUnavailable("no host bridge (LEXICHRON_BRIDGE_DIR not set)")
            job_id = self.bridge.sbatch(path)
        except (ConfigError, OSError, BridgeUnavailable, RuntimeError, TimeoutError) as exc:
            status.update(f"Submit failed: {exc}")
            self.notify("Submit failed", severity="error")
            return
        status.update(f"Submitted job {job_id}  ({path})")
        self.notify(f"Submitted job {job_id}")
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

    # -- jobs & runs ----------------------------------------------------------------
    def refresh_jobs(self) -> None:
        table = self.query_one("#jobs", DataTable)
        table.clear()
        # Job ids of queued/running jobs, or None when Slurm cannot be asked.
        self.queued_jobs: Optional[set] = None
        if self.bridge and self.bridge.available():
            try:
                rows = self.bridge.squeue(user=getpass.getuser())
                ours = self._run_job_ids()
                rows.sort(key=lambda r: 0 if r["job_id"] in ours else 1)   # lexichron jobs first
                self.queued_jobs = {r["job_id"] for r in rows}
                for row in rows:
                    table.add_row(row["job_id"], row["name"], row["state"], row["elapsed"],
                                  row["limit"], row["nodes"], row["reason"])
            except (BridgeUnavailable, TimeoutError) as exc:
                table.add_row("-", str(exc)[:40], "", "", "", "", "")
        self._refresh_runs()
        self._refresh_run_panel()

    def _run_job_ids(self) -> set:
        ids = set()
        for d in self._run_dirs():
            p = read_progress(d / "progress.json")
            if p and p.get("slurm_job_id"):
                ids.add(str(p["slurm_job_id"]))
        return ids

    def _summarize(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """summarize() plus what Slurm knows: a 'running' run whose job has
        left the queue is reported as stopped."""
        job = doc.get("slurm_job_id")
        gone = (self.queued_jobs is not None and job is not None
                and str(job) not in self.queued_jobs)
        return summarize(doc, job_gone=gone)

    def _run_dirs(self) -> List[Path]:
        proj = self.project_path
        run_root = proj.parent / ".lexichron" / "runs" if proj else None
        if not run_root or not run_root.is_dir():
            return []
        return sorted((d for d in run_root.iterdir() if (d / "progress.json").exists()), reverse=True)[:20]

    def _refresh_runs(self) -> None:
        runs = self.query_one("#runs", DataTable)
        keep = runs.cursor_row
        runs.clear()
        dirs = self._run_dirs()
        if self.selected_run not in dirs:
            self.selected_run = dirs[0] if dirs else None
        for d in dirs:
            p = read_progress(d / "progress.json")
            if not p:
                continue
            info = self._summarize(p)
            files = f"{info['done']}/{info['total']}" if info["total"] else str(info["done"])
            if info["failed"]:
                files += f" ({info['failed']} failed)"
            runs.add_row(d.name, info["state_label"], files,
                         f"{int(p.get('entries_written', 0)):,}", f"{fmt_count(info['rate'])}/s",
                         fmt_duration(info["elapsed_s"]), p.get("slurm_job_id") or "-", key=str(d))
        if dirs and self.selected_run:
            try:
                runs.move_cursor(row=dirs.index(self.selected_run), animate=False)
            except ValueError:
                pass

    @on(DataTable.RowHighlighted, "#runs")
    def _run_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key is not None and event.row_key.value:
            self.selected_run = Path(event.row_key.value)
            self._refresh_run_panel()

    def _refresh_run_panel(self) -> None:
        title = self.query_one("#run-title", Static)
        bar = self.query_one("#run-bar", ProgressBar)
        log = self.query_one("#run-log", RichLog)
        if not self.selected_run:
            if self.project_path is None:
                title.update("Fill in the corpus directory, language and n-gram size on the Settings tab; "
                             "that corpus's runs will appear here.")
            else:
                title.update("No runs yet. Use 'Run here' or 'Submit to Slurm'; runs appear here as they start.")
            for wid in ("#run-headline", "#run-detail", "#run-current", "#run-corpus", "#run-message"):
                self.query_one(wid, Static).update("")
            bar.update(total=None, progress=0)
            return
        doc = read_progress(self.selected_run / "progress.json")
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
        self.query_one("#run-detail", Static).update(info["detail"])
        self.query_one("#run-current", Static).update(info["current_text"] or " ")
        self.query_one("#run-corpus", Static).update(info["corpus_text"] or " ")
        self.query_one("#run-message", Static).update(info["message"])
        # Log tail: only rewrite when it changes, so the pane does not flicker.
        lines = tail_lines(info["log_path"], 15)
        key = (self.selected_run, tuple(lines))
        if key != self._log_shown:
            log.clear()
            if lines:
                for line in lines:
                    log.write(line)
            else:
                log.write("(no log yet)" if info["log_path"] is None else f"(cannot read {info['log_path']})")
            self._log_shown = key


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="lexichron ui", description="Terminal UI for lexichron.")
    parser.add_argument("project", nargs="?", default=None,
                        help=f"settings YAML file to open (default: the {SETTINGS_FILENAME} beside the database)")
    parser.add_argument("--stage", default="acquire", choices=sorted(STAGES))
    args = parser.parse_args(argv)
    LexichronApp(args.project, args.stage).run()
    return 0
