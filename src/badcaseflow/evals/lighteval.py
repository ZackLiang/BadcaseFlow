from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import EvalAdapter, EvalStageSpec, add_arg, find_first, make_preflight, output_dir


class LightEvalAdapter(EvalAdapter):
    backend = "lighteval"
    display_name = "LightEval"
    stages = (
        EvalStageSpec("benchmark", "LightEval benchmark", "执行 LightEval benchmark 任务", aliases=("eval",)),
    )

    def build_manifest(self, recipe: dict[str, Any], recipe_path: Path, stage: str) -> dict[str, Any]:
        launcher = find_first(recipe, ("launcher",), ("execution", "launcher")) or "accelerate"
        tasks = find_first(recipe, ("data", "tasks"), ("tasks",))
        model_args = find_first(recipe, ("model", "args"), ("model_args",))
        out = output_dir(recipe)
        command = ["lighteval", str(launcher)]
        add_arg(command, "--tasks", tasks)
        add_arg(command, "--model_args", model_args)
        add_arg(command, "--output_dir", out)
        for extra in recipe.get("args", []) if isinstance(recipe.get("args"), list) else []:
            command.append(str(extra))
        return {"execution": {"type": "command", "command": command}, "artifacts": {"output_dir": out}}

    def preflight(self, recipe: dict[str, Any], recipe_path: Path, stage: str) -> dict[str, Any]:
        base = super().preflight(recipe, recipe_path, stage)
        errors = list(base["errors"])
        warnings = list(base["warnings"])
        if not find_first(recipe, ("data", "tasks"), ("tasks",)):
            errors.append("LightEval eval requires data.tasks or tasks")
        if not find_first(recipe, ("model", "args"), ("model_args",)):
            warnings.append("LightEval model_args is not declared")
        return make_preflight(errors, warnings)
