from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class EvalStageSpec:
    stage: str
    title: str
    description: str
    aliases: tuple[str, ...] = ()


class EvalAdapter:
    backend: str = "unknown"
    display_name: str = "Unknown"
    stages: tuple[EvalStageSpec, ...] = ()

    def normalize_stage(self, recipe: dict[str, Any]) -> str:
        stage = str(
            find_first(recipe, ("stage",), ("eval", "stage"), ("evaluation", "stage")) or ""
        ).strip().lower()
        aliases = {spec.stage: spec.stage for spec in self.stages}
        for spec in self.stages:
            for alias in spec.aliases:
                aliases[alias] = spec.stage
        return aliases.get(stage, stage)

    def build_manifest(self, recipe: dict[str, Any], recipe_path: Path, stage: str) -> dict[str, Any]:
        raise NotImplementedError

    def preflight(self, recipe: dict[str, Any], recipe_path: Path, stage: str) -> dict[str, Any]:
        errors = validate_stage(stage, self.stages)
        warnings: list[str] = []
        if output_dir(recipe) is None:
            warnings.append("outputs.output_dir is not declared")
        return make_preflight(errors, warnings)


class UnknownEvalAdapter(EvalAdapter):
    def __init__(self, backend: str):
        self.backend = backend or "unknown"
        self.display_name = "Unknown"
        self.stages = ()

    def normalize_stage(self, recipe: dict[str, Any]) -> str:
        return str(find_first(recipe, ("stage",), ("eval", "stage")) or "unknown")

    def build_manifest(self, recipe: dict[str, Any], recipe_path: Path, stage: str) -> dict[str, Any]:
        return {"execution": {"type": "command", "command": ["echo", f"No eval adapter: {self.backend}"]}}

    def preflight(self, recipe: dict[str, Any], recipe_path: Path, stage: str) -> dict[str, Any]:
        return make_preflight([f"eval adapter is not implemented yet: {self.backend}"], [])


def validate_stage(stage: str, specs: Iterable[EvalStageSpec]) -> list[str]:
    supported = {spec.stage for spec in specs}
    if not stage:
        return ["eval stage is required; set top-level stage or eval.stage"]
    if stage not in supported:
        return [f"unsupported eval stage: {stage}; supported: {', '.join(sorted(supported))}"]
    return []


def make_preflight(errors: list[str], warnings: list[str]) -> dict[str, Any]:
    status = "failed" if errors else "warning" if warnings else "passed"
    return {"status": status, "errors": errors, "warnings": warnings}


def output_dir(recipe: dict[str, Any]) -> str | None:
    return find_first(recipe, ("outputs", "output_dir"), ("output_dir",), ("work_dir",))


def data_path(recipe: dict[str, Any], key: str) -> str | None:
    return find_first(recipe, ("data", key), (key,))


def find_first(recipe: dict[str, Any], *paths: tuple[str, ...]) -> Any:
    for path in paths:
        value = deep_get(recipe, path)
        if value is not None:
            return value
    return None


def deep_get(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def add_arg(command: list[str], flag: str, value: Any | None) -> None:
    if value is None:
        return
    command.extend([flag, str(value)])
