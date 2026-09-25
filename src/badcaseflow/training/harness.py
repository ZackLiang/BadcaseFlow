from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..io_utils import write_json, write_jsonl
from .diagnostics import diagnose_train_run


_METRIC_PAIR_RE = re.compile(r"([A-Za-z_][\w./-]*)=(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)", re.IGNORECASE)


# 训练框架只负责输出日志；统一 harness 负责保存状态、事件、轻量指标和失败诊断。
# 这样不同后端的运行结果可以被同一套 registry 和工作台读取。
def create_train_run_id(manifest: dict[str, Any]) -> str:
    recipe_id = str(manifest.get("recipe_id") or manifest.get("train_run_id") or "train")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"train-{_slug(recipe_id)}-{timestamp}"


def plan_train_run(manifest: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    command = _normalize_command(manifest.get("command"))
    now = datetime.now(timezone.utc).isoformat()
    artifacts = _artifact_paths()
    plan = {
        "train_run_id": run_dir.name,
        "status": "planned",
        "created_at": now,
        "updated_at": now,
        "workspace_id": manifest.get("workspace_id"),
        "recipe_id": manifest.get("recipe_id"),
        "backend": manifest.get("backend"),
        "stage": manifest.get("stage"),
        "command": command,
        "source_manifest": manifest,
        "diagnosis": diagnose_train_run(status="planned", return_code=None, stdout="", stderr=""),
        "artifacts": artifacts,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_text(run_dir / artifacts["stdout"], "")
    _write_text(run_dir / artifacts["stderr"], "")
    write_json(run_dir / "manifest.json", plan)
    write_json(run_dir / "status.json", plan)
    write_json(run_dir / artifacts["diagnosis"], plan["diagnosis"])
    write_jsonl(run_dir / "events.jsonl", [_event("planned", {"command": command})])
    write_jsonl(run_dir / "metrics.jsonl", [])
    _write_artifact_manifest(run_dir, plan)
    return plan


def run_train_manifest(
    manifest: dict[str, Any],
    run_dir: Path,
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    command = _normalize_command(manifest.get("command"))
    if not command:
        raise ValueError("train manifest command is empty")

    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()
    event_lock = threading.Lock()
    metrics_lock = threading.Lock()
    events: list[dict[str, Any]] = [
        _event("started", {"command": command, "cwd": str((cwd or Path.cwd()).resolve())}),
    ]
    metrics: list[dict[str, Any]] = []
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"

    process_env = os.environ.copy()
    if env:
        process_env.update(env)

    with stdout_path.open("w", encoding="utf-8", newline="\n") as stdout_file, stderr_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as stderr_file:
        try:
            process = subprocess.Popen(
                command,
                cwd=str(cwd.resolve()) if cwd else None,
                env=process_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except OSError as exc:
            stderr_file.write(str(exc) + "\n")
            events.append(_event("log", {"stream": "stderr", "text": str(exc)}))
            result = _build_result(
                manifest,
                run_dir,
                command,
                started_at,
                started_monotonic,
                status="failed",
                return_code=None,
            )
            result["diagnosis"] = diagnose_train_run(
                status="failed",
                return_code=None,
                stdout="",
                stderr=str(exc),
            )
            events.append(_event("finished", {"status": "failed", "return_code": None}))
            write_json(run_dir / "manifest.json", result)
            write_json(run_dir / "status.json", result)
            write_json(run_dir / result["artifacts"]["diagnosis"], result["diagnosis"])
            write_jsonl(run_dir / "events.jsonl", events)
            write_jsonl(run_dir / "metrics.jsonl", metrics)
            _write_artifact_manifest(run_dir, result)
            return result
        # stdout 和 stderr 并行读取，避免训练进程因某一侧缓冲区写满而阻塞。
        stdout_thread = threading.Thread(
            target=_copy_stream,
            args=(process.stdout, stdout_file, events, event_lock, metrics, metrics_lock, "stdout"),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_copy_stream,
            args=(process.stderr, stderr_file, events, event_lock, metrics, metrics_lock, "stderr"),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()

        timed_out = False
        try:
            return_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            return_code = process.wait()

        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)

    status = "timed_out" if timed_out else "succeeded" if return_code == 0 else "failed"
    result = _build_result(
        manifest,
        run_dir,
        command,
        started_at,
        started_monotonic,
        status=status,
        return_code=return_code,
    )
    stdout_text = _read_text(run_dir / result["artifacts"]["stdout"])
    stderr_text = _read_text(run_dir / result["artifacts"]["stderr"])
    result["diagnosis"] = diagnose_train_run(
        status=status,
        return_code=return_code,
        stdout=stdout_text,
        stderr=stderr_text,
    )
    with event_lock:
        events.append(_event("finished", {"status": status, "return_code": return_code}))
    write_json(run_dir / "manifest.json", result)
    write_json(run_dir / "status.json", result)
    write_json(run_dir / result["artifacts"]["diagnosis"], result["diagnosis"])
    write_jsonl(run_dir / "events.jsonl", events)
    write_jsonl(run_dir / "metrics.jsonl", metrics)
    _write_artifact_manifest(run_dir, result)
    return result


def parse_env_overrides(items: list[str] | None) -> dict[str, str]:
    env: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"env override must use KEY=VALUE syntax: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"env override has empty key: {item}")
        env[key] = value
    return env


def _build_result(
    manifest: dict[str, Any],
    run_dir: Path,
    command: list[str],
    started_at: datetime,
    started_monotonic: float,
    *,
    status: str,
    return_code: int | None,
) -> dict[str, Any]:
    finished_at = datetime.now(timezone.utc)
    artifacts = _artifact_paths()
    return {
        "train_run_id": run_dir.name,
        "status": status,
        "created_at": started_at.isoformat(),
        "updated_at": finished_at.isoformat(),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": round(time.monotonic() - started_monotonic, 3),
        "return_code": return_code,
        "workspace_id": manifest.get("workspace_id"),
        "recipe_id": manifest.get("recipe_id"),
        "backend": manifest.get("backend"),
        "stage": manifest.get("stage"),
        "command": command,
        "source_manifest": manifest,
        "artifacts": artifacts,
    }


def _copy_stream(
    stream: Any,
    output: Any,
    events: list[dict[str, Any]],
    event_lock: threading.Lock,
    metrics: list[dict[str, Any]],
    metrics_lock: threading.Lock,
    stream_name: str,
) -> None:
    if stream is None:
        return
    try:
        for line in stream:
            output.write(line)
            output.flush()
            stripped = line.rstrip("\n")
            if stripped:
                with event_lock:
                    events.append(_event("log", {"stream": stream_name, "text": stripped}))
                metric = parse_metric_line(stripped, stream=stream_name)
                if metric is not None:
                    with metrics_lock:
                        metrics.append(metric)
    finally:
        stream.close()


def _event(event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "time": datetime.now(timezone.utc).isoformat(),
        "event": event_type,
        **payload,
    }


def parse_metric_line(line: str, *, stream: str = "stdout") -> dict[str, Any] | None:
    lower = line.lower()
    if "metric" not in lower and "loss=" not in lower and "reward=" not in lower and "kl=" not in lower:
        return None
    # 只解析带有 metric/loss/reward/kl 线索的日志，减少普通日志误报。
    values: dict[str, float | int] = {}
    for key, raw_value in _METRIC_PAIR_RE.findall(line):
        values[key] = _parse_metric_value(raw_value)
    if not values:
        return None
    step = values.get("step") or values.get("global_step")
    return {
        "time": datetime.now(timezone.utc).isoformat(),
        "stream": stream,
        "step": step if isinstance(step, int) else None,
        "values": values,
        "raw": line,
    }


def _parse_metric_value(value: str) -> float | int:
    number = float(value)
    if number.is_integer() and "." not in value and "e" not in value.lower():
        return int(number)
    return number


def _normalize_command(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("command must be a list")
    if not all(isinstance(part, (str, int, float)) for part in value):
        raise ValueError("command list can only contain string or numeric arguments")
    return [str(part) for part in value]


def _artifact_paths() -> dict[str, str]:
    return {
        "stdout": "stdout.log",
        "stderr": "stderr.log",
        "events": "events.jsonl",
        "metrics": "metrics.jsonl",
        "status": "status.json",
        "diagnosis": "diagnosis.json",
        "artifact_manifest": "artifacts.json",
    }


def _write_artifact_manifest(run_dir: Path, result: dict[str, Any]) -> None:
    records: list[dict[str, Any]] = []
    for name, relative in sorted(result.get("artifacts", {}).items()):
        path = run_dir / relative
        records.append(
            {
                "name": name,
                "path": relative,
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else 0,
                "kind": _artifact_kind(name, relative),
            }
        )
    write_json(
        run_dir / result["artifacts"]["artifact_manifest"],
        {
            "train_run_id": result.get("train_run_id"),
            "status": result.get("status"),
            "artifacts": records,
        },
    )


def _artifact_kind(name: str, path: str) -> str:
    if name in {"stdout", "stderr"}:
        return "log"
    if path.endswith(".jsonl"):
        return "records"
    if path.endswith(".json"):
        return "metadata"
    return "file"


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value).strip("-")
