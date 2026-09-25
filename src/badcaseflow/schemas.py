from __future__ import annotations

from typing import Any


REQUIRED_TASK_FIELDS = {"sample_id", "workspace_id", "input", "expected"}
REQUIRED_TRACE_FIELDS = {"trace_id", "sample_id", "workspace_id", "steps", "final_answer"}


# schema 校验保持轻量，只检查平台必须依赖的字段；业务规则交给 eval suite。
def validate_task_sample(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = sorted(REQUIRED_TASK_FIELDS - set(record))
    if missing:
        errors.append(f"missing fields: {', '.join(missing)}")
    if "input" in record and not isinstance(record["input"], dict):
        errors.append("input must be an object")
    if "expected" in record and not isinstance(record["expected"], dict):
        errors.append("expected must be an object")
    query = record.get("input", {}).get("query") if isinstance(record.get("input"), dict) else None
    if not isinstance(query, str) or not query.strip():
        errors.append("input.query must be a non-empty string")
    criteria = record.get("expected", {}).get("success_criteria") if isinstance(record.get("expected"), dict) else None
    if criteria is not None and not isinstance(criteria, list):
        errors.append("expected.success_criteria must be a list when provided")
    return errors


def trace_for_task(task: dict[str, Any], trace: dict[str, Any]) -> bool:
    return trace.get("sample_id") == task.get("sample_id")


def validate_agent_trace(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = sorted(REQUIRED_TRACE_FIELDS - set(record))
    if missing:
        errors.append(f"missing fields: {', '.join(missing)}")
    if "steps" in record and not isinstance(record["steps"], list):
        errors.append("steps must be a list")
    if "final_answer" in record and not isinstance(record["final_answer"], str):
        errors.append("final_answer must be a string")
    if "trace_id" in record and not isinstance(record["trace_id"], str):
        errors.append("trace_id must be a string")
    if "sample_id" in record and not isinstance(record["sample_id"], str):
        errors.append("sample_id must be a string")
    return errors
