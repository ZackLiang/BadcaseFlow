from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io_utils import read_json, read_jsonl, write_json, write_jsonl


class LocalRunStore:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    def path(self, name: str) -> Path:
        return self.run_dir / name

    def init(self, workspace_id: str, source: str) -> dict[str, Any]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "run_id": self.run_dir.name,
            "workspace_id": workspace_id,
            "source": source,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "artifacts": {},
        }
        self.save_manifest(manifest)
        return manifest

    def load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"run manifest not found: {self.manifest_path}")
        return read_json(self.manifest_path)

    def save_manifest(self, manifest: dict[str, Any]) -> None:
        write_json(self.manifest_path, manifest)

    def update_artifact(self, key: str, relative_path: str) -> None:
        manifest = self.load_manifest()
        manifest.setdefault("artifacts", {})[key] = relative_path
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.save_manifest(manifest)

    def save_records(self, artifact_key: str, filename: str, records: list[dict[str, Any]]) -> Path:
        path = self.path(filename)
        write_jsonl(path, records)
        self.update_artifact(artifact_key, filename)
        return path

    def load_records(self, filename: str) -> list[dict[str, Any]]:
        path = self.path(filename)
        if not path.exists():
            raise FileNotFoundError(f"run artifact not found: {path}")
        return read_jsonl(path)

    def save_report(self, artifact_key: str, filename: str, report: dict[str, Any]) -> Path:
        path = self.path(filename)
        write_json(path, report)
        self.update_artifact(artifact_key, filename)
        return path

