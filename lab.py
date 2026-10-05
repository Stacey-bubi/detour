"""Run logging.

Every call to Run(...) creates a new folder runs/<date>_<time>_<name>/ containing:
  config.json   - the exact settings used
  code/         - a copy of the scripts that ran 
  meta.json     - time, Python version, machine, git state, plus anything added with add_meta()
  results.jsonl - one line per result, written immediately (safe if the run crashes)
  summary.md    - the short readable summary, written at the end

Past run folders are never edited. A new run always gets a new folder.
"""
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"


def _git_state():
    try:
        inside = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=ROOT,
                                capture_output=True, text=True)
        if inside.returncode != 0:
            return None  # not a git repo; the code copy in each run folder is the record
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip()
        changes = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                 capture_output=True, text=True).stdout.strip()
        return {"commit": commit or None, "uncommitted_changes": bool(changes)}
    except FileNotFoundError:
        return None


class Run:
    def __init__(self, name, config, code_files=()):
        stamp = time.strftime("%Y-%m-%d_%H%M%S")
        self.dir = RUNS / f"{stamp}_{name}"
        self.dir.mkdir(parents=True, exist_ok=False)
        (self.dir / "config.json").write_text(json.dumps(config, indent=2))
        (self.dir / "code").mkdir()
        for f in code_files:
            shutil.copy(f, self.dir / "code")
        self.meta = {"started": stamp, "python": sys.version.split()[0],
                     "machine": platform.platform(), "git": _git_state()}
        self._write_meta()
        self._results = open(self.dir / "results.jsonl", "a")
        print(f"Run folder: {self.dir.relative_to(ROOT)}")

    def add_meta(self, **items):
        self.meta.update(items)
        self._write_meta()

    def log(self, row):
        self._results.write(json.dumps(row) + "\n")
        self._results.flush()

    def finish(self, summary_text):
        self._results.close()
        self.add_meta(finished=time.strftime("%Y-%m-%d_%H%M%S"))
        (self.dir / "summary.md").write_text(summary_text)
        print(summary_text)
        print(f"\nSaved in {self.dir.relative_to(ROOT)}. Now add 1-3 lines to LOG.md.")

    def _write_meta(self):
        (self.dir / "meta.json").write_text(json.dumps(self.meta, indent=2))
