from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .base import EvalAdapter, UnknownEvalAdapter, make_preflight
from .builtin import BuiltinRuleEvalAdapter
from .command import CommandEvalAdapter
from .lighteval import LightEvalAdapter
from .opencompass import OpenCompassEvalAdapter
from .promptfoo import PromptfooEvalAdapter


_BUILTIN = BuiltinRuleEvalAdapter()
_COMMAND = CommandEvalAdapter()
_PROMPTFOO = PromptfooEvalAdapter()
_OPENCOMPASS = OpenCompassEvalAdapter()
_LIGHTEVAL = LightEvalAdapter()

_ADAPTERS: dict[str, EvalAdapter] = {
    "builtin": _BUILTIN,
    "rule": _BUILTIN,
    "badcaseflow": _BUILTIN,
    "command": _COMMAND,
    "custom": _COMMAND,
    "promptfoo": _PROMPTFOO,
    "opencompass": _OPENCOMPASS,
    "open-compass": _OPENCOMPASS,
    "lighteval": _LIGHTEVAL,
    "light-eval": _LIGHTEVAL,
}


def get_eval_adapter(backend: str | None) -> EvalAdapter:
    key = (backend or "").strip().lower()
    return _ADAPTERS.get(key, UnknownEvalAdapter(key))


def list_eval_adapters() -> list[dict[str, Any]]:
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


def build_eval_dry_run(recipe: dict[str, Any], recipe_path: Path, workspace: str | None = None) -> dict[str, Any]:
    recipe_id = recipe.get("recipe_id", recipe_path.stem)
    backend = str(recipe.get("backend", "unknown")).strip().lower()
    adapter = get_eval_adapter(backend)
    stage = adapter.normalize_stage(recipe)
    preflight = _merge_preflight(_validate_common_recipe(recipe), adapter.preflight(recipe, recipe_path, stage))
    adapter_manifest = adapter.build_manifest(recipe, recipe_path, stage)
    gate = recipe.get("gate", {}) if isinstance(recipe.get("gate"), dict) else {}
    return {
        "eval_run_id": f"dry-run-{_slug(str(recipe_id))}",
        "workspace_id": workspace,
        "recipe_path": str(recipe_path),
        "recipe_id": recipe_id,
        "backend": adapter.backend,
        "adapter": adapter.display_name,
        "stage": stage,
        "mode": recipe.get("mode", "dry_run"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "preflight": preflight,
        "execution": adapter_manifest.get("execution", {}),
        "command": (adapter_manifest.get("execution") or {}).get("command", []),
        "artifacts": adapter_manifest.get("artifacts", {}),
        "gate": {
            "min_pass_rate": float(gate.get("min_pass_rate", 0.8)),
            "allow_failed_cases": bool(gate.get("allow_failed_cases", False)),
        },
    }


def _validate_common_recipe(recipe: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    recipe_id = recipe.get("recipe_id")
    backend = recipe.get("backend")
    if recipe_id is not None and not isinstance(recipe_id, str):
        errors.append("recipe_id must be a string")
    if not isinstance(backend, str) or not backend.strip():
        errors.append("backend is required and must be a string")
    for key in ("data", "model", "outputs", "eval", "evaluation", "gate"):
        value = recipe.get(key)
        if value is not None and not isinstance(value, dict):
            errors.append(f"{key} must be a mapping when provided")
    return make_preflight(errors, warnings)


def _merge_preflight(*items: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    for item in items:
        errors.extend(item.get("errors", []))
        warnings.extend(item.get("warnings", []))
    return make_preflight(errors, warnings)


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()).strip("-").lower() or "eval"
