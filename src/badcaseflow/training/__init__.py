from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .base import TrainingAdapter, UnknownAdapter, deep_get, make_preflight, output_dir
from .custom import CustomCommandAdapter
from .llamafactory import LlamaFactoryAdapter
from .verl import VerlAdapter


_LLAMAFACTORY = LlamaFactoryAdapter()
_VERL = VerlAdapter()
_COMMAND = CustomCommandAdapter()
_ADAPTERS: dict[str, TrainingAdapter] = {
    "llamafactory": _LLAMAFACTORY,
    "llama_factory": _LLAMAFACTORY,
    "llama-factory": _LLAMAFACTORY,
    "verl": _VERL,
    "command": _COMMAND,
    "custom": _COMMAND,
}


def get_training_adapter(backend: str | None) -> TrainingAdapter:
    key = (backend or "").strip().lower()
    return _ADAPTERS.get(key, UnknownAdapter(key))


def list_training_adapters() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for adapter in _ADAPTERS.values():
        if adapter.backend in seen:
            continue
        seen.add(adapter.backend)
        for stage in adapter.stages:
            rows.append(
                {
                    "backend": adapter.backend,
                    "display_name": adapter.display_name,
                    "stage": stage.stage,
                    "title": stage.title,
                    "aliases": list(stage.aliases),
                    "description": stage.description,
                }
            )
    return rows


def build_training_dry_run(
    recipe: dict[str, Any],
    recipe_path: Path,
    workspace: str | None = None,
) -> dict[str, Any]:
    recipe_id = recipe.get("recipe_id", recipe_path.stem)
    backend = str(recipe.get("backend", "unknown")).strip().lower()
    adapter = get_training_adapter(backend)
    stage = adapter.normalize_stage(recipe)
    preflight = _merge_preflight(_validate_common_recipe(recipe), adapter.preflight(recipe, recipe_path, stage))
    command = adapter.build_command(recipe, recipe_path, stage)
    return {
        "train_run_id": f"dry-run-{_slug(str(recipe_id))}",
        "workspace_id": workspace,
        "recipe_path": str(recipe_path),
        "recipe_id": recipe_id,
        "backend": adapter.backend,
        "adapter": adapter.display_name,
        "stage": stage,
        "mode": recipe.get("mode", "dry_run"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "preflight": preflight,
        "command": command,
        "artifacts": {
            "output_dir": output_dir(recipe),
            "metrics": deep_get(recipe, ("outputs", "metrics")),
        },
    }


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value).strip("-")


def _validate_common_recipe(recipe: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    recipe_id = recipe.get("recipe_id")
    backend = recipe.get("backend")
    if recipe_id is not None and not isinstance(recipe_id, str):
        errors.append("recipe_id must be a string")
    if not isinstance(backend, str) or not backend.strip():
        errors.append("backend is required and must be a string")
    for key in ("model", "data", "training", "outputs", "rollout", "teacher", "student", "distillation"):
        value = recipe.get(key)
        if value is not None and not isinstance(value, dict):
            errors.append(f"{key} must be a mapping when provided")
    metrics = deep_get(recipe, ("outputs", "metrics"))
    if metrics is not None and not isinstance(metrics, list):
        errors.append("outputs.metrics must be a list when provided")
    return make_preflight(errors, warnings)


def _merge_preflight(*items: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    for item in items:
        errors.extend(item.get("errors", []))
        warnings.extend(item.get("warnings", []))
    return make_preflight(errors, warnings)
