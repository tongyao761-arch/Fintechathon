"""Local, immutable experiment directories and reproducibility metadata."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def prediction_hash(values) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    # Serialize first: non-finite data must not truncate an existing state file.
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.write_text(content, encoding="utf-8")


def create_run_directory(root: Path, experiment_id: str, run_id: str | None = None) -> Path:
    def validate(value):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", value):
            raise ValueError("experiment and run IDs must be simple letters, digits, '-' or '_'")
    validate(experiment_id)
    run_id = run_id or (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid4().hex[:8])
    validate(run_id)
    output = root / experiment_id / run_id
    output.mkdir(parents=True, exist_ok=False)
    return output


@contextmanager
def experiment_run(root: Path, experiment_id: str, run_id: str | None = None):
    output = create_run_directory(root, experiment_id, run_id)
    state = {"experiment_id": experiment_id, "run_id": output.name, "status": "running"}
    write_json(output / "status.json", state)
    try:
        yield output
        if not (output / "summary.json").is_file():
            raise RuntimeError("experiment finished without an accepted summary")
        state["status"] = "success"
        write_json(output / "status.json", state)
    except BaseException as exc:
        state.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        write_json(output / "status.json", state)
        raise


def provenance(root: Path, data_path: Path) -> dict:
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True, encoding="utf-8").strip()
    paths = [*root.glob("src/**/*.py"), *root.glob("scripts/*.py"),
             *root.glob("tests/*.py"), *root.glob("configs/*"),
             *root.glob("requirements*"), root / "赛题五" / "evaluate.py"]
    return {
        "data": {"path": str(data_path.relative_to(root)), "bytes": data_path.stat().st_size,
                 "sha256": sha256_file(data_path)},
        "git": {"commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
                "status": git("status", "--porcelain")},
        "source_sha256": {str(p.relative_to(root)).replace("\\", "/"): sha256_file(p)
                          for p in sorted(set(paths)) if p.is_file()},
        "dependencies": dict(sorted((d.metadata["Name"], d.version)
                                    for d in importlib.metadata.distributions())),
    }
