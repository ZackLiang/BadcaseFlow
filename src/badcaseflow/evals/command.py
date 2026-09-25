from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import EvalAdapter, EvalStageSpec, make_preflight, output_dir


class CommandEvalAdapter(EvalAdapter):
    backend = "command"
    display_name = "Custom Command Eval"
    stages = (
        EvalStageSpec(
            "custom",
            "自定义评测命令",
            "执行用户提供的评测命令，统一记录日志、指标和状态",
            aliases=("command", "shell"),
        ),
    )

    def build_manifest(self, recipe: dict[str, Any], recipe_path: Path, stage: str) -> dict[str, Any]:
        return {
            "execution": {
                "type": "command",
                "command": recipe.get("command") or [],
            },
            "artifacts": {
                "output_dir": output_dir(recipe),
                "metrics": recipe.get("outputs", {}).get("metrics") if isinstance(recipe.get("outputs"), dict) else None,
            },
        }

    def preflight(self, recipe: dict[str, Any], recipe_path: Path, stage: str) -> dict[str, Any]:
        base = super().preflight(recipe, recipe_path, stage)
        errors = list(base["errors"])
        warnings = list(base["warnings"])
        command = recipe.get("command")
        if not isinstance(command, list) or not command:
            errors.append("command backend requires a non-empty command list")
        elif not all(isinstance(part, (str, int, float)) for part in command):
            errors.append("command list can only contain string or numeric arguments")
        return make_preflight(errors, warnings)
