from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .dataset_resolver import resolve_recipe_datasets
from .io_utils import write_json
from .evals import build_eval_dry_run as build_eval_dry_run_from_recipe
from .training import build_training_dry_run


# recipe 文件保持 YAML 形式，运行前统一加载、解析数据版本并生成 manifest。
def load_simple_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"could not parse recipe YAML {path}: {exc}") from exc
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"recipe root must be a mapping: {path}")
    return value


def build_train_dry_run(
    recipe_path: Path,
    workspace: str | None = None,
    registry_root: Path | None = None,
) -> dict[str, Any]:
    recipe = load_simple_yaml(recipe_path)
    resolution = resolve_recipe_datasets(recipe, registry_root)
    manifest = build_training_dry_run(resolution["recipe"], recipe_path, workspace=workspace)
    return _attach_dataset_resolution(manifest, resolution)


def build_eval_dry_run(
    recipe_path: Path,
    workspace: str | None = None,
    registry_root: Path | None = None,
) -> dict[str, Any]:
    recipe = load_simple_yaml(recipe_path)
    resolution = resolve_recipe_datasets(recipe, registry_root)
    manifest = build_eval_dry_run_from_recipe(resolution["recipe"], recipe_path, workspace=workspace)
    return _attach_dataset_resolution(manifest, resolution)


def save_dry_run_manifest(manifest: dict[str, Any], output_root: Path) -> Path:
    run_id = manifest.get("train_run_id") or manifest.get("eval_run_id")
    path = output_root / "dry_runs" / f"{run_id}.json"
    write_json(path, manifest)
    return path


def _attach_dataset_resolution(manifest: dict[str, Any], resolution: dict[str, Any]) -> dict[str, Any]:
    manifest["dataset_resolution"] = {
        "registry_root": resolution.get("registry_root"),
        "datasets": resolution.get("datasets", []),
    }
    preflight = manifest.get("preflight") or {}
    errors = list(preflight.get("errors", []))
    warnings = list(preflight.get("warnings", []))
    errors.extend(resolution.get("errors", []))
    warnings.extend(resolution.get("warnings", []))
    manifest["preflight"] = _make_preflight(errors, warnings)
    return manifest


def _make_preflight(errors: list[str], warnings: list[str]) -> dict[str, Any]:
    if errors:
        status = "failed"
    elif warnings:
        status = "warning"
    else:
        status = "passed"
    return {"status": status, "errors": errors, "warnings": warnings}
