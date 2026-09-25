from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io_utils import read_json, write_json
from .iteration import render_iteration_plan_markdown


ACTION_STATUSES = {"proposed", "accepted", "in_progress", "done", "rejected", "blocked"}


def list_actions(plan_path: Path, *, status: str | None = None) -> list[dict[str, Any]]:
    plan = read_json(plan_path)
    actions = _actions(plan)
    if status:
        _validate_status(status)
        actions = [action for action in actions if action.get("status") == status]
    return actions


def update_action(
    plan_path: Path,
    *,
    action_id: str,
    status: str | None = None,
    owner: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    if not action_id.strip():
        raise ValueError("action_id is required")
    if status:
        _validate_status(status)

    plan = read_json(plan_path)
    actions = _actions(plan)
    action = _find_action(actions, action_id)
    now = _now()
    changes: dict[str, Any] = {}

    if status and action.get("status") != status:
        changes["status"] = {"from": action.get("status"), "to": status}
        action["status"] = status
    if owner is not None and action.get("owner") != owner:
        changes["owner"] = {"from": action.get("owner"), "to": owner}
        action["owner"] = owner
    if note:
        note_record = {"time": now, "text": note, "status": action.get("status"), "owner": action.get("owner")}
        action.setdefault("notes", []).append(note_record)
        changes["note"] = note

    if changes:
        action["updated_at"] = now
        action.setdefault("history", []).append({"time": now, "event": "updated", "changes": changes})
        plan["updated_at"] = now
        plan["summary"] = {**(plan.get("summary") or {}), "action_status_counts": _status_counts(actions)}
        write_json(plan_path, plan)
        markdown_path = plan_path.with_suffix(".md")
        if markdown_path.exists():
            markdown_path.write_text(render_iteration_plan_markdown(plan), encoding="utf-8", newline="\n")
    return action


def _actions(plan: dict[str, Any]) -> list[dict[str, Any]]:
    actions = plan.get("actions")
    if not isinstance(actions, list):
        raise ValueError("iteration plan has no actions list")
    return [action for action in actions if isinstance(action, dict)]


def _find_action(actions: list[dict[str, Any]], action_id: str) -> dict[str, Any]:
    for action in actions:
        if action.get("action_id") == action_id:
            return action
    raise KeyError(f"action not found: {action_id}")


def _validate_status(status: str) -> None:
    if status not in ACTION_STATUSES:
        raise ValueError(f"status must be one of: {', '.join(sorted(ACTION_STATUSES))}")


def _status_counts(actions: list[dict[str, Any]]) -> dict[str, int]:
    counts = {status: 0 for status in sorted(ACTION_STATUSES)}
    for action in actions:
        status = str(action.get("status") or "proposed")
        counts[status] = counts.get(status, 0) + 1
    return {status: count for status, count in counts.items() if count}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
