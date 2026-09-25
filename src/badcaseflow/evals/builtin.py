from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import EvalAdapter, EvalStageSpec, data_path, make_preflight, output_dir


class BuiltinRuleEvalAdapter(EvalAdapter):
    backend = "builtin"
    display_name = "BadcaseFlow Builtin Rule Eval"
    stages = (
        EvalStageSpec(
            "agent_trace",
            "Agent Trace 规则评测",
            "使用 BadcaseFlow 内置规则评测 Agent trace 和 eval suite",
            aliases=("rule", "rules", "trace"),
        ),
    )

    def build_manifest(self, recipe: dict[str, Any], recipe_path: Path, stage: str) -> dict[str, Any]:
        return {
            "execution": {
                "type": "builtin_rule",
                "traces": data_path(recipe, "traces"),
                "suite": data_path(recipe, "suite"),
            },
            "artifacts": {
                "output_dir": output_dir(recipe),
                "metrics": ["eval_report"],
            },
        }

    def preflight(self, recipe: dict[str, Any], recipe_path: Path, stage: str) -> dict[str, Any]:
        base = super().preflight(recipe, recipe_path, stage)
        errors = list(base["errors"])
        warnings = list(base["warnings"])
        traces = data_path(recipe, "traces")
        suite = data_path(recipe, "suite")
        if not traces:
            errors.append("data.traces is required for builtin eval")
        elif not Path(str(traces)).exists():
            warnings.append(f"trace file does not exist yet: {traces}")
        if not suite:
            errors.append("data.suite is required for builtin eval")
        elif not Path(str(suite)).exists():
            warnings.append(f"eval suite does not exist yet: {suite}")
        return make_preflight(errors, warnings)
