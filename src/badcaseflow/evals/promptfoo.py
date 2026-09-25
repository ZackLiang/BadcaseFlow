from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import EvalAdapter, EvalStageSpec, add_arg, data_path, make_preflight, output_dir


class PromptfooEvalAdapter(EvalAdapter):
    backend = "promptfoo"
    display_name = "promptfoo"
    stages = (
        EvalStageSpec("suite", "promptfoo suite", "执行 promptfoo eval 配置", aliases=("eval", "test")),
    )

    def build_manifest(self, recipe: dict[str, Any], recipe_path: Path, stage: str) -> dict[str, Any]:
        config = data_path(recipe, "config") or recipe.get("config")
        out = output_dir(recipe)
        command = ["promptfoo", "eval"]
        add_arg(command, "--config", config)
        if out:
            add_arg(command, "--output", str(Path(str(out)) / "promptfoo-results.json"))
        for extra in recipe.get("args", []) if isinstance(recipe.get("args"), list) else []:
            command.append(str(extra))
        return {"execution": {"type": "command", "command": command}, "artifacts": {"output_dir": out}}

    def preflight(self, recipe: dict[str, Any], recipe_path: Path, stage: str) -> dict[str, Any]:
        base = super().preflight(recipe, recipe_path, stage)
        errors = list(base["errors"])
        warnings = list(base["warnings"])
        config = data_path(recipe, "config") or recipe.get("config")
        if not config:
            errors.append("promptfoo eval requires data.config or config")
        elif not Path(str(config)).exists():
            warnings.append(f"promptfoo config does not exist yet: {config}")
        return make_preflight(errors, warnings)
