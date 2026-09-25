from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from .registry import get_dataset


# recipe 可以只保存 dataset_version_id，执行前统一解析成真实文件路径。
def resolve_recipe_datasets(recipe: dict[str, Any], registry_root: Path | None) -> dict[str, Any]:
    resolved = copy.deepcopy(recipe)
    result: dict[str, Any] = {
        "registry_root": str(registry_root) if registry_root is not None else None,
        "datasets": [],
        "errors": [],
        "warnings": [],
        "recipe": resolved,
    }
    if registry_root is None:
        return result

    data = resolved.setdefault("data", {})
    if not isinstance(data, dict):
        result["errors"].append("data must be a mapping before resolving dataset versions")
        return result

    # 不同训练/评测后端对数据字段命名不同，这里集中维护映射关系。
    _resolve_key(
        data,
        registry_root,
        result,
        key="dataset_version_id",
        target=None,
        plural=False,
    )
    _resolve_key(
        data,
        registry_root,
        result,
        key="dataset_version_ids",
        target="train_files",
        plural=True,
    )
    _resolve_key(
        data,
        registry_root,
        result,
        key="train_dataset_version_id",
        target="train_file",
        plural=False,
    )
    _resolve_key(
        data,
        registry_root,
        result,
        key="train_dataset_version_ids",
        target="train_files",
        plural=True,
    )
    _resolve_key(
        data,
        registry_root,
        result,
        key="val_dataset_version_id",
        target="val_file",
        plural=False,
    )
    _resolve_key(
        data,
        registry_root,
        result,
        key="val_dataset_version_ids",
        target="val_files",
        plural=True,
    )
    _resolve_key(
        data,
        registry_root,
        result,
        key="eval_dataset_version_id",
        target="eval_file",
        plural=False,
    )
    _resolve_key(
        data,
        registry_root,
        result,
        key="trace_dataset_version_id",
        target="traces",
        plural=False,
    )
    _resolve_key(
        data,
        registry_root,
        result,
        key="suite_dataset_version_id",
        target="suite",
        plural=False,
    )
    return result


def _resolve_key(
    data: dict[str, Any],
    registry_root: Path,
    result: dict[str, Any],
    *,
    key: str,
    target: str | None,
    plural: bool,
) -> None:
    raw = data.get(key)
    if raw in (None, ""):
        return
    values = raw if isinstance(raw, list) else [raw]
    if not all(isinstance(value, str) and value.strip() for value in values):
        result["errors"].append(f"data.{key} must be a non-empty string or list of strings")
        return

    resolved_paths: list[str] = []
    for dataset_version_id in values:
        try:
            dataset = get_dataset(registry_root, dataset_version_id)
        except KeyError:
            result["errors"].append(f"dataset version not found: {dataset_version_id}")
            continue
        path = str(dataset.get("path") or "")
        if not path:
            result["errors"].append(f"dataset version has no path: {dataset_version_id}")
            continue
        resolved_paths.append(path)
        result["datasets"].append(
            {
                "field": key,
                "target": target or _target_for_kind(str(dataset.get("kind") or ""), plural=plural),
                "dataset_version_id": dataset.get("dataset_version_id"),
                "dataset_id": dataset.get("dataset_id"),
                "kind": dataset.get("kind"),
                "path": path,
                "record_count": dataset.get("record_count"),
                "schema_status": dataset.get("schema_status"),
                "sha256": dataset.get("sha256"),
            }
        )
        if dataset.get("schema_status") != "passed":
            result["warnings"].append(f"dataset schema is not passed: {dataset.get('dataset_version_id')}")

    if not resolved_paths:
        return
    resolved_target = target or _target_for_kind(str(result["datasets"][-1].get("kind") or ""), plural=plural)
    if resolved_target is None:
        result["warnings"].append(f"data.{key} was resolved but no target field is known")
        return
    if resolved_target in data and data[resolved_target] not in (None, "", []):
        result["warnings"].append(f"data.{resolved_target} already exists; dataset version did not overwrite it")
        return
    data[resolved_target] = resolved_paths if plural or len(resolved_paths) > 1 else resolved_paths[0]


def _target_for_kind(kind: str, *, plural: bool) -> str | None:
    if kind == "eval_suite":
        return "suites" if plural else "suite"
    if kind == "trace":
        return "traces"
    if kind in {"task", "sft", "preference", "rl_prompt"}:
        return "train_files" if plural else "train_file"
    return None
