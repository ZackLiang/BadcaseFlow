from __future__ import annotations

import os
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..evaluation import evaluate_traces
from ..io_utils import read_jsonl, write_json, write_jsonl
from ..training.harness import parse_metric_line


# 评测 harness 与训练 harness 使用相同的 run 目录结构，方便工作台统一读取。
def create_eval_run_id(manifest: dict[str, Any]) -> str:
    recipe_id = str(manifest.get("recipe_id") or manifest.get("eval_run_id") or "eval")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"eval-{_slug(recipe_id)}-{timestamp}"


def plan_eval_run(manifest: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    now = _now()
    artifacts = _artifact_paths()
    result = {
        "eval_run_id": run_dir.name,
        "status": "planned",
        "quality_status": "unknown",
        "created_at": now,
        "updated_at": now,
        "workspace_id": manifest.get("workspace_id"),
        "recipe_id": manifest.get("recipe_id"),
        "backend": manifest.get("backend"),
        "stage": manifest.get("stage"),
        "execution": manifest.get("execution", {}),
        "command": manifest.get("command", []),
        "gate": manifest.get("gate", {}),
        "source_manifest": manifest,
        "artifacts": artifacts,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_text(run_dir / artifacts["stdout"], "")
    _write_text(run_dir / artifacts["stderr"], "")
    write_json(run_dir / "manifest.json", result)
    write_json(run_dir / "status.json", result)
    write_jsonl(run_dir / "events.jsonl", [_event("planned", {"execution": result["execution"]})])
    write_jsonl(run_dir / "metrics.jsonl", [])
    _write_artifact_manifest(run_dir, result)
    return result


def run_eval_manifest(
    manifest: dict[str, Any],
    run_dir: Path,
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    execution = manifest.get("execution") or {}
    execution_type = execution.get("type")
    if execution_type == "builtin_rule":
        return _run_builtin_eval(manifest, run_dir)
    return _run_command_eval(manifest, run_dir, cwd=cwd, env=env, timeout_seconds=timeout_seconds)


def _run_builtin_eval(manifest: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()
    execution = manifest.get("execution") or {}
    traces = read_jsonl(Path(str(execution.get("traces"))))
    suite = read_jsonl(Path(str(execution.get("suite"))))
    results, report, accepted, rejected = evaluate_traces(traces, suite)
    artifacts = _artifact_paths()
    write_jsonl(run_dir / artifacts["eval_results"], results)
    write_json(run_dir / artifacts["eval_report"], report)
    write_jsonl(run_dir / artifacts["accepted"], accepted)
    write_jsonl(run_dir / artifacts["rejected"], rejected)
    _write_text(run_dir / artifacts["stdout"], "")
    _write_text(run_dir / artifacts["stderr"], "")
    metrics = [_metric_from_report(report)]
    status = _build_result(manifest, run_dir, started_at, started_monotonic, status="succeeded", return_code=0)
    status["eval_report"] = _report_summary(report)
    status["quality_status"] = _quality_status(report, manifest.get("gate", {}))
    status["artifacts"].update(
        {
            "eval_report": artifacts["eval_report"],
            "eval_results": artifacts["eval_results"],
            "accepted": artifacts["accepted"],
            "rejected": artifacts["rejected"],
        }
    )
    write_json(run_dir / "manifest.json", status)
    write_json(run_dir / "status.json", status)
    write_jsonl(run_dir / "events.jsonl", [_event("started", {"execution": execution}), _event("finished", status)])
    write_jsonl(run_dir / "metrics.jsonl", metrics)
    _write_artifact_manifest(run_dir, status)
    return status


def _run_command_eval(
    manifest: dict[str, Any],
    run_dir: Path,
    *,
    cwd: Path | None,
    env: dict[str, str] | None,
    timeout_seconds: int | None,
) -> dict[str, Any]:
    command = _normalize_command((manifest.get("execution") or {}).get("command") or manifest.get("command"))
    if not command:
        raise ValueError("eval manifest command is empty")

    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()
    events: list[dict[str, Any]] = [_event("started", {"command": command, "cwd": str((cwd or Path.cwd()).resolve())})]
    metrics: list[dict[str, Any]] = []
    event_lock = threading.Lock()
    metrics_lock = threading.Lock()
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
            result = _build_result(manifest, run_dir, started_at, started_monotonic, status="failed", return_code=None)
            result["quality_status"] = "unknown"
            events.append(_event("finished", {"status": "failed", "return_code": None}))
            write_json(run_dir / "manifest.json", result)
            write_json(run_dir / "status.json", result)
            write_jsonl(run_dir / "events.jsonl", events)
            write_jsonl(run_dir / "metrics.jsonl", metrics)
            _write_artifact_manifest(run_dir, result)
            return result

        # 两条输出流并行消费，避免外部评测命令因缓冲区写满而卡住。
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
    result = _build_result(manifest, run_dir, started_at, started_monotonic, status=status, return_code=return_code)
    result["quality_status"] = _quality_status_from_metrics(metrics, manifest.get("gate", {}))
    with event_lock:
        events.append(_event("finished", {"status": status, "return_code": return_code}))
    write_json(run_dir / "manifest.json", result)
    write_json(run_dir / "status.json", result)
    write_jsonl(run_dir / "events.jsonl", events)
    write_jsonl(run_dir / "metrics.jsonl", metrics)
    _write_artifact_manifest(run_dir, result)
    return result


def _build_result(
    manifest: dict[str, Any],
    run_dir: Path,
    started_at: datetime,
    started_monotonic: float,
    *,
    status: str,
    return_code: int | None,
) -> dict[str, Any]:
    finished_at = datetime.now(timezone.utc)
    return {
        "eval_run_id": run_dir.name,
        "status": status,
        "quality_status": "unknown",
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
        "execution": manifest.get("execution", {}),
        "command": manifest.get("command", []),
        "gate": manifest.get("gate", {}),
        "source_manifest": manifest,
        "artifacts": _artifact_paths(),
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


def _quality_status(report: dict[str, Any], gate: dict[str, Any]) -> str:
    pass_rate = float(report.get("pass_rate", 0.0) or 0.0)
    failed = int(report.get("failed", 0) or 0)
    min_pass_rate = float(gate.get("min_pass_rate", 0.8) or 0.8)
    allow_failed_cases = bool(gate.get("allow_failed_cases", False))
    if pass_rate < min_pass_rate:
        return "blocked"
    if failed > 0 and not allow_failed_cases:
        return "blocked"
    return "approved"


def _quality_status_from_metrics(metrics: list[dict[str, Any]], gate: dict[str, Any]) -> str:
    latest: dict[str, Any] = {}
    for metric in metrics:
        values = metric.get("values")
        if isinstance(values, dict):
            latest.update(values)
    if "pass_rate" not in latest and "failed" not in latest:
        return "unknown"
    report = {"pass_rate": latest.get("pass_rate", 0.0), "failed": latest.get("failed", 0)}
    return _quality_status(report, gate)


def _metric_from_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "time": _now(),
        "stream": "report",
        "step": None,
        "values": _report_summary(report),
        "raw": "eval_report",
    }


def _report_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "total": int(report.get("total", 0) or 0),
        "passed": int(report.get("passed", 0) or 0),
        "failed": int(report.get("failed", 0) or 0),
        "pass_rate": float(report.get("pass_rate", 0.0) or 0.0),
    }


def _artifact_paths() -> dict[str, str]:
    return {
        "stdout": "stdout.log",
        "stderr": "stderr.log",
        "events": "events.jsonl",
        "metrics": "metrics.jsonl",
        "status": "status.json",
        "artifact_manifest": "artifacts.json",
        "eval_report": "eval_report.json",
        "eval_results": "eval_results.jsonl",
        "accepted": "accepted.jsonl",
        "rejected": "rejected.jsonl",
    }


def _write_artifact_manifest(run_dir: Path, result: dict[str, Any]) -> None:
    records = []
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
    write_json(run_dir / result["artifacts"]["artifact_manifest"], {"eval_run_id": result.get("eval_run_id"), "artifacts": records})


def _artifact_kind(name: str, path: str) -> str:
    if name in {"stdout", "stderr"}:
        return "log"
    if path.endswith(".jsonl"):
        return "records"
    if path.endswith(".json"):
        return "metadata"
    return "file"


def _normalize_command(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("command must be a list")
    if not all(isinstance(part, (str, int, float)) for part in value):
        raise ValueError("command list can only contain string or numeric arguments")
    return [str(part) for part in value]


def _event(event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {"time": _now(), "event": event_type, **payload}


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value).strip("-").lower()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
