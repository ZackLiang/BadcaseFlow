from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .registry import get_model


# 部署模块只生成可检查的 runbook，不直接启动服务或切换流量。
SERVING_BACKENDS = ("vllm", "sglang", "openai-compatible")
DEPLOYMENT_MODES = ("plan", "runbook")


def build_deployment_plan(
    root: Path,
    *,
    model_id: str,
    backend: str = "vllm",
    mode: str = "plan",
    deployment_id: str | None = None,
    served_model_name: str | None = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    python: str = "python",
    endpoint: str | None = None,
    canary_percent: float = 5.0,
    min_pass_rate: float | None = None,
    previous_model_id: str | None = None,
    rollback_endpoint: str | None = None,
    allow_blocked: bool = False,
) -> dict[str, Any]:
    backend = backend.strip().lower()
    mode = mode.strip().lower()
    if backend not in SERVING_BACKENDS:
        raise ValueError(f"backend must be one of: {', '.join(SERVING_BACKENDS)}")
    if mode not in DEPLOYMENT_MODES:
        raise ValueError(f"mode must be one of: {', '.join(DEPLOYMENT_MODES)}")
    if port <= 0:
        raise ValueError("port must be positive")
    if canary_percent < 0 or canary_percent > 100:
        raise ValueError("canary_percent must be between 0 and 100")

    model = get_model(root, model_id)
    if model.get("status") != "approved" and not allow_blocked:
        raise ValueError(f"model is not approved: {model.get('status')}")

    resolved_deployment_id = deployment_id or _deployment_id(model_id)
    served_name = served_model_name or model_id
    commands = _commands_for_backend(
        backend=backend,
        model_path=str(model.get("model_path") or ""),
        served_model_name=served_name,
        host=host,
        port=port,
        python=python,
        endpoint=endpoint,
    )
    health_url = endpoint or f"http://{host if host != '0.0.0.0' else '127.0.0.1'}:{port}/v1/models"
    plan = {
        "version": 1,
        "deployment_id": resolved_deployment_id,
        "model_id": model_id,
        "model_status": model.get("status"),
        "model_path": model.get("model_path"),
        "backend": backend,
        "mode": mode,
        "served_model_name": served_name,
        "created_at": _now(),
        "status": "planned",
        "service": {
            "host": host,
            "port": port,
            "endpoint": endpoint or f"http://{host if host != '0.0.0.0' else '127.0.0.1'}:{port}/v1",
            "health_url": health_url,
        },
        "gate": {
            "model_status_required": "approved" if not allow_blocked else "recorded",
            "latest_eval_report": model.get("latest_eval_report", {}),
            "promotion_reasons": model.get("promotion_reasons", []),
            "min_pass_rate": min_pass_rate,
        },
        "canary": {
            "percent": canary_percent,
            "checks": ["health", "sample_eval", "latency", "error_rate"],
            "advance_when": {
                "manual_review": True,
                "pass_rate_at_least": min_pass_rate,
                "no_new_blocker": True,
            },
        },
        "rollback": {
            "previous_model_id": previous_model_id,
            "endpoint": rollback_endpoint,
            "actions": ["stop_candidate", "restore_previous_endpoint", "rerun_smoke_eval"],
        },
        "commands": commands,
    }
    return plan


def render_deployment_script(plan: dict[str, Any]) -> str:
    lines = [
        "# BadcaseFlow deployment plan",
        "$ErrorActionPreference = 'Stop'",
        "",
        f"# deployment_id={plan.get('deployment_id')}",
        f"# model_id={plan.get('model_id')}",
        "",
    ]
    for group in ("prepare", "serve", "verify", "rollback"):
        commands = plan.get("commands", {}).get(group, [])
        if not commands:
            continue
        lines.append(f"# {group}")
        lines.extend(str(command) for command in commands)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _commands_for_backend(
    *,
    backend: str,
    model_path: str,
    served_model_name: str,
    host: str,
    port: int,
    python: str,
    endpoint: str | None,
) -> dict[str, list[str]]:
    if backend == "vllm":
        serve = [
            _local_command(
                [
                    python,
                    "-m",
                    "vllm.entrypoints.openai.api_server",
                    "--model",
                    model_path,
                    "--served-model-name",
                    served_model_name,
                    "--host",
                    host,
                    "--port",
                    str(port),
                ]
            )
        ]
    elif backend == "sglang":
        serve = [
            _local_command(
                [
                    python,
                    "-m",
                    "sglang.launch_server",
                    "--model-path",
                    model_path,
                    "--served-model-name",
                    served_model_name,
                    "--host",
                    host,
                    "--port",
                    str(port),
                ]
            )
        ]
    else:
        serve = [f"# use existing OpenAI-compatible endpoint: {endpoint or 'http://127.0.0.1:8000/v1'}"]

    base_endpoint = endpoint or f"http://{host if host != '0.0.0.0' else '127.0.0.1'}:{port}/v1"
    return {
        "prepare": ["# confirm model artifact path and runtime dependencies before serving"],
        "serve": serve,
        "verify": [
            f"curl {base_endpoint}/models",
            "# run BadcaseFlow eval smoke before increasing canary traffic",
        ],
        "rollback": [
            "# stop candidate service",
            "# restore previous endpoint or route traffic back to the previous model",
        ],
    }


def _deployment_id(model_id: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    slug = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in model_id).strip("-").lower()
    return f"deploy-{slug or 'model'}-{timestamp}"


def _local_command(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
