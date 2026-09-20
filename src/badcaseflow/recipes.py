from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io_utils import write_json


def load_simple_yaml(path: Path) -> dict[str, Any]:
    lines = _load_yaml_lines(path)
    value, index = _parse_map(lines, 0, 0)
    if index != len(lines):
        raise ValueError(f"could not parse all recipe lines in {path}")
    return value


def build_train_dry_run(recipe_path: Path, workspace: str | None = None) -> dict[str, Any]:
    recipe = load_simple_yaml(recipe_path)
    recipe_id = recipe.get("recipe_id", recipe_path.stem)
    backend = recipe.get("backend", "unknown")
    preflight = _preflight(recipe, recipe_path.parent)
    command = _command_for_recipe(recipe, recipe_path)
    return {
        "train_run_id": f"dry-run-{_slug(str(recipe_id))}",
        "workspace_id": workspace,
        "recipe_path": str(recipe_path),
        "recipe_id": recipe_id,
        "backend": backend,
        "mode": recipe.get("mode", "dry_run"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "preflight": preflight,
        "command": command,
        "artifacts": {
            "output_dir": _deep_get(recipe, ["outputs", "output_dir"]),
            "metrics": _deep_get(recipe, ["outputs", "metrics"]),
        },
    }


def save_dry_run_manifest(manifest: dict[str, Any], output_root: Path) -> Path:
    run_id = manifest["train_run_id"]
    path = output_root / "dry_runs" / f"{run_id}.json"
    write_json(path, manifest)
    return path


def _preflight(recipe: dict[str, Any], recipe_dir: Path) -> dict[str, Any]:
    warnings: list[str] = []
    errors: list[str] = []
    data = recipe.get("data", {})
    for key in ("train_file", "val_file"):
        value = data.get(key)
        if not value:
            if key == "train_file":
                errors.append("data.train_file is required")
            continue
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = (recipe_dir / ".." / ".." / candidate).resolve()
        if not candidate.exists():
            warnings.append(f"{key} does not exist yet: {value}")
    backend = recipe.get("backend")
    if backend not in {"llamafactory", "verl"}:
        warnings.append(f"backend adapter is not implemented yet: {backend}")
    return {"status": "failed" if errors else "warning" if warnings else "passed", "errors": errors, "warnings": warnings}


def _command_for_recipe(recipe: dict[str, Any], recipe_path: Path) -> list[str]:
    backend = recipe.get("backend")
    if backend == "llamafactory":
        return ["llamafactory-cli", "train", str(recipe_path)]
    if backend == "verl":
        data = recipe.get("data", {})
        student = recipe.get("student", {})
        teacher = recipe.get("teacher", {})
        return [
            "python",
            "-m",
            "verl.trainer.main_distill",
            f"data.train_files={data.get('train_file')}",
            f"data.val_files={data.get('val_file')}",
            f"student.model.path={student.get('model')}",
            f"teacher.model.path={teacher.get('model')}",
        ]
    return ["echo", "No adapter command is available for this backend yet."]


def _load_yaml_lines(path: Path) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        rows.append((indent, raw.strip()))
    return rows


def _parse_map(lines: list[tuple[int, str]], index: int, indent: int) -> tuple[dict[str, Any], int]:
    result: dict[str, Any] = {}
    while index < len(lines):
        line_indent, content = lines[index]
        if line_indent < indent:
            break
        if line_indent > indent:
            raise ValueError(f"unexpected indentation near: {content}")
        if content.startswith("- "):
            raise ValueError(f"unexpected list item near: {content}")
        key, raw_value = _split_key_value(content)
        if raw_value == "":
            next_index = index + 1
            if next_index < len(lines) and lines[next_index][0] > line_indent and lines[next_index][1].startswith("- "):
                value, index = _parse_list(lines, next_index, lines[next_index][0])
            else:
                value, index = _parse_map(lines, next_index, line_indent + 2)
            result[key] = value
        else:
            result[key] = _parse_scalar(raw_value)
            index += 1
    return result, index


def _parse_list(lines: list[tuple[int, str]], index: int, indent: int) -> tuple[list[Any], int]:
    result: list[Any] = []
    while index < len(lines):
        line_indent, content = lines[index]
        if line_indent < indent:
            break
        if line_indent != indent or not content.startswith("- "):
            raise ValueError(f"unexpected list syntax near: {content}")
        result.append(_parse_scalar(content[2:].strip()))
        index += 1
    return result, index


def _split_key_value(content: str) -> tuple[str, str]:
    if ":" not in content:
        raise ValueError(f"expected key: value syntax near: {content}")
    key, value = content.split(":", 1)
    return key.strip(), value.strip()


def _parse_scalar(value: str) -> Any:
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip('"').strip("'")


def _deep_get(value: dict[str, Any], path: list[str]) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value).strip("-")

