"""``lexichron ui``: a terminal interface for configuring, running and
submitting pipeline stages from a project file.

Layout: a tab per concern.
  Project  — form for the project file (generated from the stage's signature)
             and, beside it, the resolved call exactly as --dry-run prints it
  Run      — run the stage here (login node test, or inside an allocation)
  Submit   — Slurm resources; writes the batch script and submits it through
             the host bridge when one is running
  Jobs     — squeue for your jobs, plus progress.json of the latest runs
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
    Button, DataTable, Footer, Header, Input, Label, RichLog, Select, Static,
    Switch, TabbedContent, TabPane,
)

from lexichron import __version__
from lexichron.bridge import BridgeUnavailable, HostBridge, default_bridge_dir
from lexichron.cli import STAGES, _format_call, _resolve
from lexichron.config import ConfigError, build_call, load_project
from lexichron.ui.schema import Field, SLURM_FIELDS, Section, stage_sections
from lexichron.ui.slurm import write_sbatch
from ngramprep.ngram_acquire.progress import read_progress

__all__ = ["LexichronApp", "main"]


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


# ---------------------------------------------------------------------------
# The app
# ---------------------------------------------------------------------------

class LexichronApp(App):
    TITLE = f"lexichron {__version__}"
    CSS = """
    #form { width: 1fr; padding: 0 1; }
    #preview { width: 1fr; padding: 0 1; border-left: solid $secondary; }
    .section { text-style: bold; color: $accent; margin-top: 1; }
    .field-help { color: $text-muted; margin-bottom: 1; }
    .row { height: auto; }
    .row Label { width: 24; padding-top: 1; }
    .row Input, .row Select { width: 1fr; }
    #call { padding: 1; }
    #status { color: $warning; padding: 0 1; height: auto; }
    .actions { height: auto; padding: 1; }
    .actions Button { margin-right: 2; }
    #runlog, #joblog { height: 1fr; }
    """
    BINDINGS = [("ctrl+s", "save", "Save"), ("ctrl+q", "quit", "Quit")]

    def __init__(self, project_path: Optional[str | os.PathLike] = None, stage: str = "acquire") -> None:
        super().__init__()
        # The project file is optional. Without one, it defaults to
        # <db_path_stub>/project.yaml once that field is filled in, so a user
        # who only wants to download a corpus never has to think about it.
        self.explicit_project: Optional[Path] = Path(project_path).resolve() if project_path else None
        self.stage = stage
        self.func = _resolve(STAGES[stage][0])
        self.sections: List[Section] = stage_sections(self.func, stage) + [SLURM_FIELDS]
        self.config: Dict[str, Any] = {}
        if self.explicit_project and self.explicit_project.exists():
            self.config = load_project(self.explicit_project)
        bridge_dir = default_bridge_dir()
        self.bridge: Optional[HostBridge] = HostBridge(bridge_dir) if bridge_dir else None
        self.proc: Optional[subprocess.Popen] = None

    @property
    def project_path(self) -> Optional[Path]:
        """Where the project file is (or will be) saved."""
        try:
            typed = self.query_one("#project-path", Input).value.strip()
        except Exception:
            typed = ""
        if typed:
            return Path(typed).expanduser().resolve()
        if self.explicit_project:
            return self.explicit_project
        stub = (self.config.get("corpus") or {}).get("db_path_stub")
        if stub:
            return Path(str(stub)).expanduser().resolve() / "project.yaml"
        return None

    def _require_project_path(self) -> Path:
        p = self.project_path
        if p is None:
            raise ConfigError("set corpus.db_path_stub (or a project file path) first")
        return p

    # -- layout -----------------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(initial="tab-project"):
            with TabPane("Project", id="tab-project"):
                with Horizontal():
                    with VerticalScroll(id="form"):
                        with Horizontal(classes="row"):
                            yield Label("project file")
                            yield Input(value=str(self.explicit_project or ""), id="project-path",
                                        placeholder="(optional; defaults to <db_path_stub>/project.yaml)")
                        for sec in self.sections:
                            if sec.name == "slurm":
                                continue
                            yield Static(f"[{sec.name}]", classes="section")
                            for f in sec.fields:
                                yield from self._field_widgets(sec.name, f)
                    with Vertical(id="preview"):
                        yield Static("Resolved call", classes="section")
                        yield Static("", id="call")
                        yield Static("", id="status")
                        with Horizontal(classes="actions"):
                            yield Button("Save project file", id="save", variant="primary")
            with TabPane("Run", id="tab-run"):
                with Horizontal(classes="actions"):
                    yield Button("Run here", id="run", variant="success")
                    yield Button("Stop", id="stop", variant="error")
                yield RichLog(id="runlog", wrap=True, highlight=False, markup=False)
            with TabPane("Submit", id="tab-submit"):
                with VerticalScroll():
                    yield Static("[slurm]", classes="section")
                    for f in SLURM_FIELDS.fields:
                        yield from self._field_widgets("slurm", f)
                    with Horizontal(classes="actions"):
                        yield Button("Write batch script", id="write-sbatch")
                        yield Button("Submit to Slurm", id="submit", variant="primary")
                    yield Static("", id="submit-status")
            with TabPane("Jobs", id="tab-jobs"):
                with Horizontal(classes="actions"):
                    yield Button("Refresh", id="refresh-jobs")
                    yield Button("Cancel selected", id="cancel-job", variant="error")
                yield DataTable(id="jobs")
                yield Static("Latest runs", classes="section")
                yield DataTable(id="runs")
        yield Footer()

    def _field_widgets(self, section: str, f: Field):
        wid = f"f-{section}-{f.name}"
        current = (self.config.get(section) or {}).get(f.name, f.default)
        label = f.name + (" *" if f.required else "")
        with Horizontal(classes="row"):
            yield Label(label)
            if f.kind == "bool":
                yield Switch(value=bool(current), id=wid)
            elif f.kind == "choice" and f.choices:
                opts = [(c, c) for c in f.choices]
                val = str(current) if current is not None else Select.NULL
                if val is not Select.NULL and val not in f.choices:
                    opts.append((val, val))
                yield Select(opts, value=val, allow_blank=True, id=wid)
            else:
                yield Input(value=_to_widget_text(current), id=wid,
                            placeholder=f"({f.kind}{', optional' if not f.required else ''})")
        if f.help:
            yield Static(f.help, classes="field-help")

    def on_mount(self) -> None:
        jobs = self.query_one("#jobs", DataTable)
        jobs.add_columns("Job", "Name", "State", "Elapsed", "Limit", "Nodes", "Reason")
        jobs.cursor_type = "row"
        runs = self.query_one("#runs", DataTable)
        runs.add_columns("Run", "State", "Files", "Entries", "Updated", "Message")
        self._refresh_preview()
        self.refresh_jobs()
        self.set_interval(10, self.refresh_jobs)
        if not self.bridge or not self.bridge.available():
            self.query_one("#submit-status", Static).update(
                "No host bridge: 'Submit' and the job table need the UI started through "
                "the project's .venv/lexichron-ui launcher. 'Write batch script' still works.")

    # -- form -> config -----------------------------------------------------------
    def _collect(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {}
        for sec in self.sections:
            vals: Dict[str, Any] = {}
            for f in sec.fields:
                w = self.query_one(f"#f-{sec.name}-{f.name}")
                try:
                    v = _from_widget(f, w.value)
                except ValueError:
                    raise ConfigError(f"{sec.name}.{f.name}: not a valid {f.kind}")
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
        dest_line = f"# project file: {dest if dest else '(set corpus.db_path_stub)'}"
        try:
            kwargs = build_call(self.func, {k: v for k, v in cfg.items() if k != "slurm"}, self.stage)
        except ConfigError as exc:
            call.update(f"# (fill in the required settings)\n{dest_line}")
            status.update(str(exc))
            return
        call.update(_format_call(self.func, kwargs) + "\n\n" + dest_line)
        status.update("")

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
            fh.write(f"# lexichron project file (written by lexichron ui {__version__})\n")
            yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=False)
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
        if self.bridge and self.bridge.available():
            try:
                for row in self.bridge.squeue(user=getpass.getuser()):
                    table.add_row(row["job_id"], row["name"], row["state"], row["elapsed"],
                                  row["limit"], row["nodes"], row["reason"])
            except (BridgeUnavailable, TimeoutError) as exc:
                table.add_row("-", str(exc)[:40], "", "", "", "", "")
        runs = self.query_one("#runs", DataTable)
        runs.clear()
        proj = self.project_path
        run_root = proj.parent / ".lexichron" / "runs" if proj else None
        if run_root and run_root.is_dir():
            for d in sorted(run_root.iterdir(), reverse=True)[:10]:
                p = read_progress(d / "progress.json")
                if not p:
                    continue
                files = f"{p.get('files_done', 0)}/{p.get('files_total', 0)}"
                if p.get("files_failed"):
                    files += f" ({p['files_failed']} failed)"
                runs.add_row(d.name, p.get("state", ""), files, f"{p.get('entries_written', 0):,}",
                             (p.get("updated") or "")[11:19], (p.get("message") or "")[:50])


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="lexichron ui", description="Terminal UI for lexichron.")
    parser.add_argument("project", nargs="?", default=None,
                        help="project YAML file (optional; defaults to <db_path_stub>/project.yaml)")
    parser.add_argument("--stage", default="acquire", choices=sorted(STAGES))
    args = parser.parse_args(argv)
    LexichronApp(args.project, args.stage).run()
    return 0
