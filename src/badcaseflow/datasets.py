from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .io_utils import write_json
from .schemas import validate_agent_trace, validate_task_sample


# 数据集模块只做本地文件级检查：格式、记录数量、摘要和基础 schema。
# 复杂的数据质量规则由具体数据工厂或评测 adapter 负责。
SUPPORTED_DATASET_KINDS = (
    "task",
    "trace",
    "eval_suite",
    "sft",
    "preference",
    "rl_prompt",
    "eval_result",
    "eval_report",
    "generic_jsonl",
    "generic_json",
)


JSONL_KINDS = {
    "task",
    "trace",
    "eval_suite",
    "sft",
    "preference",
    "rl_prompt",
    "eval_result",
    "generic_jsonl",
}

JSON_KINDS = {"eval_report", "generic_json"}

RecordValidator = Callable[[dict[str, Any]], list[str]]


def build_dataset_version(
    path: Path,
    *,
    kind: str,
    dataset_id: str | None = None,
    workspace_id: str | None = None,
    description: str | None = None,
    source: dict[str, Any] | None = None,
    max_errors: int = 20,
) -> dict[str, Any]:
    if kind not in SUPPORTED_DATASET_KINDS:
        raise ValueError(f"unsupported dataset kind: {kind}")
    if not path.exists():
        raise FileNotFoundError(f"dataset path not found: {path}")
    if not path.is_file():
        raise ValueError(f"dataset path must be a file: {path}")

    digest = file_sha256(path)
    logical_id = _slug(dataset_id or path.stem)
    file_format = _detect_format(path, kind)
    # JSONL 可以逐行返回错误；parquet 先读取 metadata，避免一次性加载整个数据集。
    if file_format == "jsonl":
        profile = _profile_jsonl(path, kind=kind, max_errors=max_errors)
    elif file_format == "parquet":
        profile = _profile_parquet(path)
    else:
        profile = _profile_json(path, kind=kind, max_errors=max_errors)

    schema_errors = profile["schema_errors"]
    parse_errors = profile["parse_errors"]
    schema_status = "failed" if schema_errors or parse_errors else "passed"
    version_id = f"dataset-{logical_id}-{digest[:12]}"
    stat = path.stat()
    manifest = {
        "version": 1,
        "dataset_version_id": version_id,
        "dataset_id": logical_id,
        "kind": kind,
        "file_format": file_format,
        "path": str(path),
        "name": path.name,
        "workspace_id": workspace_id,
        "description": description,
        "source": source or {},
        "bytes": stat.st_size,
        "sha256": digest,
        "record_count": profile["record_count"],
        "schema_status": schema_status,
        "schema_errors": schema_errors,
        "parse_errors": parse_errors,
        "sample_ids": profile.get("sample_ids", [])[:20],
        "created_at": _now(),
    }
    return manifest


def write_dataset_manifest(path: Path, manifest: dict[str, Any]) -> Path:
    write_json(path, manifest)
    return path


def write_dataset_collection(path: Path, datasets: list[dict[str, Any]]) -> Path:
    write_json(
        path,
        {
            "version": 1,
            "created_at": _now(),
            "datasets": datasets,
            "summary": {
                "total": len(datasets),
                "passed": sum(1 for item in datasets if item.get("schema_status") == "passed"),
                "failed": sum(1 for item in datasets if item.get("schema_status") == "failed"),
            },
        },
    )
    return path


def dataset_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_version_id": manifest.get("dataset_version_id"),
        "dataset_id": manifest.get("dataset_id"),
        "kind": manifest.get("kind"),
        "path": manifest.get("path"),
        "bytes": manifest.get("bytes", 0),
        "sha256": manifest.get("sha256"),
        "record_count": manifest.get("record_count", 0),
        "schema_status": manifest.get("schema_status"),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _profile_jsonl(path: Path, *, kind: str, max_errors: int) -> dict[str, Any]:
    validator = _validator_for(kind)
    parse_errors: list[dict[str, Any]] = []
    schema_errors: list[dict[str, Any]] = []
    sample_ids: list[str] = []
    record_count = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                parse_errors.append({"line": line_no, "message": str(exc)})
                if len(parse_errors) >= max_errors:
                    break
                continue
            if not isinstance(record, dict):
                schema_errors.append({"line": line_no, "errors": ["record must be a JSON object"]})
                if len(schema_errors) >= max_errors:
                    break
                continue
            record_count += 1
            sample_id = record.get("sample_id")
            if isinstance(sample_id, str) and sample_id not in sample_ids:
                sample_ids.append(sample_id)
            errors = validator(record) if validator is not None else []
            if errors:
                schema_errors.append({"line": line_no, "errors": errors})
                if len(schema_errors) >= max_errors:
                    break
    return {
        "record_count": record_count,
        "schema_errors": schema_errors,
        "parse_errors": parse_errors,
        "sample_ids": sample_ids,
    }


def _profile_json(path: Path, *, kind: str, max_errors: int) -> dict[str, Any]:
    parse_errors: list[dict[str, Any]] = []
    schema_errors: list[dict[str, Any]] = []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"record_count": 0, "schema_errors": [], "parse_errors": [{"line": exc.lineno, "message": str(exc)}]}
    if not isinstance(value, dict):
        schema_errors.append({"line": 1, "errors": ["JSON dataset must be an object"]})
    elif kind == "eval_report":
        errors = _validate_eval_report(value)
        if errors:
            schema_errors.append({"line": 1, "errors": errors[:max_errors]})
    return {"record_count": 1 if isinstance(value, dict) else 0, "schema_errors": schema_errors, "parse_errors": parse_errors}


def _profile_parquet(path: Path) -> dict[str, Any]:
    # pyarrow 不是基础依赖，缺少时把问题记录到 manifest，而不是导入阶段直接崩溃。
    try:
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except ImportError as exc:
        return {
            "record_count": 0,
            "schema_errors": [],
            "parse_errors": [{"line": 1, "message": f"parquet metadata needs pyarrow: {exc}"}],
            "sample_ids": [],
        }
    try:
        metadata = pq.read_metadata(path)
    except Exception as exc:  # pragma: no cover - depends on pyarrow parser details
        return {
            "record_count": 0,
            "schema_errors": [],
            "parse_errors": [{"line": 1, "message": str(exc)}],
            "sample_ids": [],
        }
    return {
        "record_count": metadata.num_rows,
        "schema_errors": [],
        "parse_errors": [],
        "sample_ids": [],
    }


def _detect_format(path: Path, kind: str) -> str:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return "parquet"
    if kind in JSON_KINDS:
        return "json"
    if kind in JSONL_KINDS:
        return "jsonl"
    if suffix == ".json":
        return "json"
    return "jsonl"


def _validator_for(kind: str) -> RecordValidator | None:
    return {
        "task": validate_task_sample,
        "trace": validate_agent_trace,
        "eval_suite": _validate_eval_suite_record,
        "sft": _validate_sft_record,
        "preference": _validate_preference_record,
        "rl_prompt": _validate_rl_prompt_record,
        "eval_result": _validate_eval_result_record,
    }.get(kind)


def _validate_eval_suite_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in ("eval_item_id", "sample_id", "suite_id", "checks"):
        if field not in record:
            errors.append(f"missing fields: {field}")
    if "checks" in record and not isinstance(record["checks"], list):
        errors.append("checks must be a list")
    for index, check in enumerate(record.get("checks", []) if isinstance(record.get("checks"), list) else []):
        if not isinstance(check, dict):
            errors.append(f"checks[{index}] must be an object")
            continue
        if not isinstance(check.get("name"), str) or not check.get("name"):
            errors.append(f"checks[{index}].name must be a non-empty string")
        if not isinstance(check.get("type"), str) or not check.get("type"):
            errors.append(f"checks[{index}].type must be a non-empty string")
        if "weight" in check and not isinstance(check["weight"], (int, float)):
            errors.append(f"checks[{index}].weight must be a number")
    return errors


def _validate_sft_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        errors.append("messages must be a non-empty list")
        return errors
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            errors.append(f"messages[{index}] must be an object")
            continue
        if message.get("role") not in {"system", "user", "assistant", "tool"}:
            errors.append(f"messages[{index}].role is not supported")
        if not isinstance(message.get("content"), str):
            errors.append(f"messages[{index}].content must be a string")
    return errors


def _validate_preference_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in ("prompt", "chosen", "rejected"):
        if not isinstance(record.get(field), str) or not record.get(field):
            errors.append(f"{field} must be a non-empty string")
    return errors


def _validate_rl_prompt_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    prompt = record.get("prompt")
    if not isinstance(prompt, (str, list)) or prompt == "":
        errors.append("prompt must be a non-empty string or message list")
    if "reward_model" in record and not isinstance(record["reward_model"], dict):
        errors.append("reward_model must be an object when provided")
    return errors


def _validate_eval_result_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in ("eval_id", "sample_id", "suite_id", "passed"):
        if field not in record:
            errors.append(f"missing fields: {field}")
    if "passed" in record and not isinstance(record["passed"], bool):
        errors.append("passed must be a boolean")
    if "scores" in record and not isinstance(record["scores"], dict):
        errors.append("scores must be an object")
    if "failure_types" in record and not isinstance(record["failure_types"], list):
        errors.append("failure_types must be a list")
    return errors


def _validate_eval_report(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in ("total", "passed", "failed", "pass_rate"):
        if field not in record:
            errors.append(f"missing fields: {field}")
    for field in ("total", "passed", "failed"):
        if field in record and not isinstance(record[field], int):
            errors.append(f"{field} must be an integer")
    if "pass_rate" in record and not isinstance(record["pass_rate"], (int, float)):
        errors.append("pass_rate must be a number")
    return errors


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()).strip("-").lower()
    return slug or "dataset"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
