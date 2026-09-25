from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import EvalAdapter, EvalStageSpec, add_arg, data_path, make_preflight, output_dir


class OpenCompassEvalAdapter(EvalAdapter):
    backend = "opencompass"
    display_name = "OpenCompass"
    stages = (
        EvalStageSpec("benchmark", "OpenCompass benchmark", "执行 OpenCompass benchmark 配置", aliases=("eval",)),
    )

    def build_manifest(self, recipe: dict[str, Any], recipe_path: Path, stage: str) -> dict[str, Any]:
        config = data_path(recipe, "config") or recipe.get("config")
        out = output_dir(recipe)
        command = ["opencompass"]
        if config:
            command.append(str(config))
        add_arg(command, "--work-dir", out)
        for extra in recipe.get("args", []) if isinstance(recipe.get("args"), list) else []:
            command.append(str(extra))
        return {"execution": {"type": "command", "command": command}, "artifacts": {"output_dir": out}}

    def preflight(self, recipe: dict[str, Any], recipe_path: Path, stage: str) -> dict[str, Any]:
        base = super().preflight(recipe, recipe_path, stage)
        errors = list(base["errors"])
        warnings = list(base["warnings"])
        config = data_path(recipe, "config") or recipe.get("config")
        if not config:
            errors.append("OpenCompass eval requires data.config or config")
        elif not Path(str(config)).exists():
            warnings.append(f"OpenCompass config does not exist yet: {config}")
        return make_preflight(errors, warnings)
