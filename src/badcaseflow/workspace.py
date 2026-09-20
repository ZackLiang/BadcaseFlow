from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io_utils import read_json, write_json


DEFAULT_CONFIG_DIR = ".badcaseflow"


def init_workspace(root: Path, workspace_id: str, force: bool = False) -> dict[str, Any]:
    config_dir = root / DEFAULT_CONFIG_DIR
    config_dir.mkdir(parents=True, exist_ok=True)
    workspace_path = config_dir / "workspace.json"
    remotes_path = config_dir / "remotes.json"
    if workspace_path.exists() and not force:
        raise FileExistsError(f"workspace config already exists: {workspace_path}")
    workspace = {
        "workspace_id": workspace_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "default_run_root": "runs",
        "default_artifact_root": "artifacts",
        "default_examples": {
            "seed_tasks": "examples/datasets/seed_tasks.jsonl",
            "eval_suite": "examples/datasets/eval_tasks.jsonl",
            "sft_recipe": "examples/recipes/sft_llamafactory.example.yaml",
            "opd_recipe": "examples/recipes/opd_verl.example.yaml",
        },
    }
    write_json(workspace_path, workspace)
    if not remotes_path.exists():
        write_json(remotes_path, {"remotes": {}})
    return workspace


def load_workspace(root: Path) -> dict[str, Any]:
    path = root / DEFAULT_CONFIG_DIR / "workspace.json"
    if not path.exists():
        raise FileNotFoundError(f"workspace config not found: {path}")
    return read_json(path)


def remotes_path(root: Path) -> Path:
    return root / DEFAULT_CONFIG_DIR / "remotes.json"

