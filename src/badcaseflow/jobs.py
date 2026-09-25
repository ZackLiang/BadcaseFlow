from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io_utils import read_json, write_json


JOB_STATUSES = {"planned", "running", "succeeded", "failed", "timed_out"}
LOG_STREAMS = {"stdout", "stderr", "events"}
COMMAND_GROUPS = ("prepare", "sync", "run", "collect")


# job 是 launch plan 的状态外壳，不改变底层训练或评测命令。
# 这样本地执行、SSH 计划和后续队列调度可以共用同一套状态文件。
def create_job_id(plan: dict[str, Any]) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"job-{_slug(str(plan.get('plan_id') or 'launch'))}-{timestamp}"


def submit_job(
    *,
    plan_path: Path,
    jobs_root: Path = Path("runs/jobs"),
    job_id: str | None = None,
    execute: bool = False,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    plan = read_json(plan_path)
    resolved_job_id = job_id or create_job_id(plan)
    job_dir = jobs_root / resolved_job_id
    if job_dir.exists():
        raise FileExistsError(f"job already exists: {job_dir}")
    job_dir.mkdir(parents=True, exist_ok=False)
    write_json(job_dir / "launch_plan.json", plan)
    (job_dir / "stdout.log").write_text("", encoding="utf-8", newline="\n")
    (job_dir / "stderr.log").write_text("", encoding="utf-8", newline="\n")
    (job_dir / "events.jsonl").write_text("", encoding="utf-8", newline="\n")

    now = _now()
    status = {
        "job_id": resolved_job_id,
        "status": "planned",
        "created_at": now,
        "updated_at": now,
        "plan_path": str(plan_path),
        "job_dir": str(job_dir),
        "plan_id": plan.get("plan_id"),
        "kind": plan.get("kind"),
        "launcher": plan.get("launcher"),
        "mode": plan.get("mode"),
        "target": plan.get("target"),
        "execute": execute,
        "commands_total": _count_commands(plan),
        "commands_completed": 0,
        "commands_failed": 0,
        "command_results": [],
        "artifacts": {
            "launch_plan": "launch_plan.json",
            "status": "status.json",
            "events": "events.jsonl",
            "stdout": "stdout.log",
            "stderr": "stderr.log",
        },
    }
    _append_event(job_dir, "job_created", {"plan_id": plan.get("plan_id"), "execute": execute})
    write_json(job_dir / "status.json", status)
    if not execute:
        return status

    return execute_job(job_dir, timeout_seconds=timeout_seconds)


def execute_job(job_dir: Path, *, timeout_seconds: int | None = None) -> dict[str, Any]:
    status = read_json(job_dir / "status.json")
    plan = read_json(job_dir / "launch_plan.json")
    if status.get("status") not in {"planned", "failed", "timed_out"}:
        raise ValueError(f"job cannot be executed from status: {status.get('status')}")

    started_at = _now()
    status.update({"status": "running", "started_at": started_at, "updated_at": started_at})
    write_json(job_dir / "status.json", status)
    _append_event(job_dir, "job_started", {"timeout_seconds": timeout_seconds})

    commands_completed = 0
    commands_failed = 0
    results: list[dict[str, Any]] = list(status.get("command_results") or [])
    final_status = "succeeded"
    for group in COMMAND_GROUPS:
        for command in plan.get("commands", {}).get(group, []) or []:
            result = _run_command(job_dir, group=group, command=str(command), timeout_seconds=timeout_seconds, plan=plan)
            results.append(result)
            if result["status"] == "succeeded":
                commands_completed += 1
            else:
                commands_failed += 1
                final_status = result["status"]
                break
        if final_status != "succeeded":
            break

    finished_at = _now()
    status.update(
        {
            "status": final_status,
            "updated_at": finished_at,
            "finished_at": finished_at,
            "commands_completed": commands_completed,
            "commands_failed": commands_failed,
            "command_results": results,
            "return_code": results[-1].get("return_code") if results else None,
        }
    )
    write_json(job_dir / "status.json", status)
    _append_event(
        job_dir,
        "job_finished",
        {
            "status": final_status,
            "commands_completed": commands_completed,
            "commands_failed": commands_failed,
        },
    )
    return status


def collect_job(job_dir: Path, *, timeout_seconds: int | None = None) -> dict[str, Any]:
    status = read_json(job_dir / "status.json")
    plan = read_json(job_dir / "launch_plan.json")
    results: list[dict[str, Any]] = list(status.get("command_results") or [])
    commands = plan.get("commands", {}).get("collect", []) or []
    if not commands:
        status["collection_status"] = "skipped"
        status["updated_at"] = _now()
        write_json(job_dir / "status.json", status)
        _append_event(job_dir, "collect_skipped", {"reason": "no collect commands"})
        return status

    _append_event(job_dir, "collect_started", {"commands": len(commands)})
    collection_status = "succeeded"
    for command in commands:
        result = _run_command(job_dir, group="collect", command=str(command), timeout_seconds=timeout_seconds, plan=plan)
        results.append(result)
        if result["status"] != "succeeded":
            collection_status = result["status"]
            break
    status.update(
        {
            "collection_status": collection_status,
            "updated_at": _now(),
            "command_results": results,
        }
    )
    write_json(job_dir / "status.json", status)
    _append_event(job_dir, "collect_finished", {"status": collection_status})
    return status


def get_job_status(job_dir: Path) -> dict[str, Any]:
    return read_json(job_dir / "status.json")


def list_jobs(jobs_root: Path) -> list[dict[str, Any]]:
    if not jobs_root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for child in jobs_root.iterdir():
        if not child.is_dir():
            continue
        status_path = child / "status.json"
        if status_path.exists():
            rows.append(read_json(status_path))
    return sorted(rows, key=lambda item: str(item.get("updated_at", "")), reverse=True)


def read_job_log(job_dir: Path, *, stream: str = "stdout", tail: int | None = None) -> list[str]:
    if stream not in LOG_STREAMS:
        raise ValueError(f"stream must be one of: {', '.join(sorted(LOG_STREAMS))}")
    path = job_dir / ("events.jsonl" if stream == "events" else f"{stream}.log")
    if not path.exists():
        raise FileNotFoundError(f"log not found: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    if tail is not None and tail >= 0:
        return lines[-tail:]
    return lines


def _run_command(
    job_dir: Path,
    *,
    group: str,
    command: str,
    timeout_seconds: int | None,
    plan: dict[str, Any],
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()
    _append_event(job_dir, "command_started", {"group": group, "command": command})
    cwd = _command_cwd(plan)
    with (job_dir / "stdout.log").open("a", encoding="utf-8", newline="\n") as stdout_file, (
        job_dir / "stderr.log"
    ).open("a", encoding="utf-8", newline="\n") as stderr_file:
        try:
            process = subprocess.Popen(
                command,
                shell=True,
                cwd=str(cwd) if cwd else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            try:
                stdout, stderr = process.communicate(timeout=timeout_seconds)
                return_code = process.returncode
                status = "succeeded" if return_code == 0 else "failed"
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                return_code = process.returncode
                status = "timed_out"
        except OSError as exc:
            stdout = ""
            stderr = str(exc) + "\n"
            return_code = None
            status = "failed"

        if stdout:
            stdout_file.write(stdout)
            stdout_file.flush()
        if stderr:
            stderr_file.write(stderr)
            stderr_file.flush()

    finished_at = datetime.now(timezone.utc)
    result = {
        "group": group,
        "command": command,
        "status": status,
        "return_code": return_code,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": round(time.monotonic() - started_monotonic, 3),
    }
    _append_event(job_dir, "command_finished", result)
    return result


def _command_cwd(plan: dict[str, Any]) -> Path | None:
    if plan.get("launcher") != "local":
        return None
    cwd = plan.get("cwd")
    return Path(str(cwd)).resolve() if cwd else None


def _count_commands(plan: dict[str, Any]) -> int:
    return sum(len(plan.get("commands", {}).get(group, []) or []) for group in COMMAND_GROUPS)


def _append_event(job_dir: Path, event_type: str, payload: dict[str, Any]) -> None:
    event = {"time": _now(), "event": event_type, **payload}
    with (job_dir / "events.jsonl").open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(event, ensure_ascii=False, sort_keys=True))
        f.write("\n")


def _slug(value: str) -> str:
    import re

    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-").lower()
    return slug or "job"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
