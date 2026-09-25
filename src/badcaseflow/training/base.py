from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


# 训练 recipe 只描述用户意图，adapter 负责把它翻译成具体框架命令。
@dataclass(frozen=True)
class StageSpec:
    stage: str
    title: str
    description: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class AdapterSummary:
    backend: str
    display_name: str
    stages: tuple[StageSpec, ...]


class TrainingAdapter:
    backend: str
    display_name: str
    stages: tuple[StageSpec, ...]
    stage_aliases: dict[str, str]

    def normalize_stage(self, recipe: dict[str, Any]) -> str:
        raw_stage = find_first(
            recipe,
            ("stage",),
            ("training", "stage"),
            ("method", "stage"),
            ("distillation", "stage"),
        )
        if raw_stage is None:
            return "unknown"
        stage = str(raw_stage).strip().lower()
        return self.stage_aliases.get(stage, stage)

    def summary(self) -> AdapterSummary:
        return AdapterSummary(self.backend, self.display_name, self.stages)

    def build_command(self, recipe: dict[str, Any], recipe_path: Path, stage: str) -> list[str]:
        raise NotImplementedError

    def preflight(self, recipe: dict[str, Any], recipe_path: Path, stage: str) -> dict[str, Any]:
        raise NotImplementedError


class UnknownAdapter(TrainingAdapter):
    def __init__(self, backend: str) -> None:
        self.backend = backend
        self.display_name = backend or "unknown"
        self.stages = ()
        self.stage_aliases = {}

    def build_command(self, recipe: dict[str, Any], recipe_path: Path, stage: str) -> list[str]:
        return ["echo", f"No training adapter is available for backend: {self.backend}"]

    def preflight(self, recipe: dict[str, Any], recipe_path: Path, stage: str) -> dict[str, Any]:
        return make_preflight(
            [f"training adapter is not implemented yet: {self.backend or 'unknown'}"],
            [],
        )


def make_preflight(errors: Iterable[str], warnings: Iterable[str]) -> dict[str, Any]:
    error_list = list(errors)
    warning_list = list(warnings)
    if error_list:
        status = "failed"
    elif warning_list:
        status = "warning"
    else:
        status = "passed"
    return {"status": status, "errors": error_list, "warnings": warning_list}


def deep_get(value: dict[str, Any], path: Iterable[str]) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def find_first(recipe: dict[str, Any], *paths: tuple[str, ...]) -> Any:
    for path in paths:
        value = deep_get(recipe, path)
        if value not in (None, ""):
            return value
    return None


def collect_stage_warnings(adapter: TrainingAdapter, recipe: dict[str, Any], stage: str) -> list[str]:
    warnings: list[str] = []
    declared = [
        str(value).strip().lower()
        for value in (
            deep_get(recipe, ("stage",)),
            deep_get(recipe, ("training", "stage")),
            deep_get(recipe, ("method", "stage")),
            deep_get(recipe, ("distillation", "stage")),
        )
        if value not in (None, "")
    ]
    normalized = {adapter.stage_aliases.get(value, value) for value in declared}
    if len(normalized) > 1:
        warnings.append(f"recipe declares multiple training stages: {', '.join(sorted(normalized))}")
    return warnings


def collect_stage_errors(adapter: TrainingAdapter, stage: str) -> list[str]:
    supported = {item.stage for item in adapter.stages}
    if stage in supported:
        return []
    if stage == "unknown":
        return ["training stage is required; set top-level stage or training.stage"]
    return [f"{adapter.backend} adapter does not support stage: {stage}"]


def collect_data_path_warnings(recipe: dict[str, Any], recipe_path: Path) -> list[str]:
    warnings: list[str] = []
    data = recipe.get("data", {})
    if not isinstance(data, dict):
        return warnings
    for key in ("train_file", "val_file", "eval_file", "path"):
        value = data.get(key)
        if value in (None, ""):
            continue
        for item in ensure_list(value):
            if not isinstance(item, str):
                continue
            if not _path_exists(item, recipe_path):
                warnings.append(f"data.{key} does not exist yet: {item}")
    return warnings


def ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def hydra_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, list):
        rendered = [_quote_hydra_string(item) if isinstance(item, str) else hydra_value(item) for item in value]
        return "[" + ",".join(rendered) + "]"
    if value is None:
        return "null"
    return str(value)


def hydra_file_list(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, list):
        return "[" + ",".join(_quote_hydra_string(str(item)) for item in value) + "]"
    text = str(value)
    if text.startswith("[") and text.endswith("]"):
        return text
    return "[" + _quote_hydra_string(text) + "]"


def add_override(command: list[str], key: str, value: Any, *, list_value: bool = False) -> None:
    if value in (None, ""):
        return
    rendered = hydra_file_list(value) if list_value else hydra_value(value)
    if rendered is not None:
        command.append(f"{key}={rendered}")


def append_user_overrides(command: list[str], recipe: dict[str, Any]) -> None:
    # 同时支持列表形式和嵌套字典形式，方便保留目标框架原生参数。
    overrides = recipe.get("overrides") or recipe.get("hydra_overrides")
    if isinstance(overrides, list):
        command.extend(str(item) for item in overrides if item not in (None, ""))
    elif isinstance(overrides, dict):
        for key, value in flatten_dict(overrides):
            add_override(command, key, value)


def flatten_dict(value: dict[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    result: list[tuple[str, Any]] = []
    for key, item in value.items():
        joined = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict):
            result.extend(flatten_dict(item, joined))
        else:
            result.append((joined, item))
    return result


def first_model_path(recipe: dict[str, Any]) -> Any:
    return find_first(
        recipe,
        ("model", "path"),
        ("model", "base_model"),
        ("student", "model"),
        ("model_name_or_path",),
        ("actor_rollout_ref", "model", "path"),
    )


def output_dir(recipe: dict[str, Any]) -> Any:
    return find_first(recipe, ("outputs", "output_dir"), ("output_dir",), ("trainer", "default_local_dir"))


def _quote_hydra_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def _path_exists(value: str, recipe_path: Path) -> bool:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.exists()
    search_roots = (Path.cwd(), recipe_path.parent, recipe_path.parent / ".." / "..")
    return any((root / candidate).resolve().exists() for root in search_roots)
