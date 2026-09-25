from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import StageSpec, TrainingAdapter, collect_stage_errors, collect_stage_warnings, make_preflight


class CustomCommandAdapter(TrainingAdapter):
    backend = "command"
    display_name = "Custom Command"
    stages = (
        StageSpec("custom", "自定义命令", "用于接入暂未内置的训练脚本或 CI smoke harness"),
    )
    stage_aliases = {"shell": "custom", "script": "custom"}

    def build_command(self, recipe: dict[str, Any], recipe_path: Path, stage: str) -> list[str]:
        command = recipe.get("command")
        if isinstance(command, list):
            return [str(part) for part in command]
        return []

    def preflight(self, recipe: dict[str, Any], recipe_path: Path, stage: str) -> dict[str, Any]:
        errors = collect_stage_errors(self, stage)
        warnings = collect_stage_warnings(self, recipe, stage)
        command = recipe.get("command")
        if not isinstance(command, list) or not command:
            errors.append("command backend requires a non-empty command list")
        elif not all(isinstance(part, (str, int, float)) for part in command):
            errors.append("command list can only contain string or numeric arguments")
        return make_preflight(errors, warnings)
