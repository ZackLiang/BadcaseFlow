from __future__ import annotations

from typing import Any


_PATTERNS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    (
        "CUDA_OOM",
        ("cuda out of memory", "torch.cuda.outofmemoryerror", "cublas_status_alloc_failed"),
        "降低 batch size、序列长度或显存占用；检查 rollout 并发和 ZeRO/FSDP/offload 配置。",
    ),
    (
        "GPU_UNAVAILABLE",
        ("no cuda gpus are available", "cuda is not available", "nvidia-smi", "driver/library version mismatch"),
        "检查 CUDA 驱动、容器 GPU 挂载、torch/cuda 版本和机器资源。",
    ),
    (
        "RAY_RESOURCE",
        ("ray", "resources", "no available node types", "cannot schedule", "pending forever"),
        "检查 Ray 集群资源、GPU 数量、placement group 和 trainer/rollout/teacher 资源池配置。",
    ),
    (
        "DEPENDENCY_MISSING",
        ("modulenotfounderror", "importerror", "no module named", "cannot import name"),
        "安装缺失依赖，或确认当前 Python 环境与训练框架环境一致。",
    ),
    (
        "COMMAND_NOT_FOUND",
        ("the system cannot find the file specified", "no such file or directory", "not recognized as"),
        "检查 launcher 命令是否在 PATH 中，例如 python、llamafactory-cli、torchrun 或 uv。",
    ),
    (
        "DATA_FILE_NOT_FOUND",
        ("filenotfounderror", "file not found", "no such file"),
        "检查 recipe 中 data.train_file/data.val_file 路径和远程同步路径。",
    ),
    (
        "DATA_SCHEMA",
        ("keyerror", "jsondecodeerror", "schema", "parquet", "column"),
        "检查训练数据格式、字段名、parquet/jsonl schema 和 adapter 期望格式。",
    ),
    (
        "PERMISSION_DENIED",
        ("permission denied", "access is denied", "operation not permitted"),
        "检查输出目录、缓存目录、模型目录和远程工作目录权限。",
    ),
    (
        "CONFIG_ERROR",
        ("omegaconf", "hydra", "missing mandatory value", "config key", "unknown argument"),
        "检查 recipe overrides、Hydra 参数名和目标训练框架版本。",
    ),
)


def diagnose_train_run(
    *,
    status: str,
    return_code: int | None,
    stdout: str,
    stderr: str,
) -> dict[str, Any]:
    text = f"{stderr}\n{stdout}".lower()
    if status == "planned":
        return _diagnosis("PLANNED", "info", "训练 run 已创建但尚未执行。", [], [])
    if status == "succeeded":
        return _diagnosis("SUCCEEDED", "info", "训练命令执行成功。", [], [])
    if status == "timed_out":
        return _diagnosis(
            "TIMEOUT",
            "error",
            "训练命令超时后被终止。",
            _evidence_lines(stdout, stderr, ("timeout", "timed out")),
            ["增大 timeout，或检查训练是否卡在数据读取、Ray 调度、模型下载或分布式初始化。"],
        )

    for category, patterns, suggestion in _PATTERNS:
        evidence = _evidence_lines(stdout, stderr, patterns)
        if evidence:
            return _diagnosis(category, "error", f"训练失败，疑似原因：{category}。", evidence, [suggestion])

    fallback = f"训练命令失败，退出码：{return_code}。" if return_code is not None else "训练命令启动失败。"
    return _diagnosis(
        "PROCESS_FAILED",
        "error",
        fallback,
        _tail_evidence(stdout, stderr),
        ["查看 stderr.log/stdout.log 中靠近末尾的错误堆栈，并补充诊断规则。"],
    )


def _diagnosis(
    category: str,
    severity: str,
    summary: str,
    evidence: list[str],
    suggestions: list[str],
) -> dict[str, Any]:
    return {
        "category": category,
        "severity": severity,
        "summary": summary,
        "evidence": evidence[:8],
        "suggestions": suggestions,
    }


def _evidence_lines(stdout: str, stderr: str, patterns: tuple[str, ...]) -> list[str]:
    evidence: list[str] = []
    for source, text in (("stderr", stderr), ("stdout", stdout)):
        for line in text.splitlines():
            lowered = line.lower()
            if any(pattern in lowered for pattern in patterns):
                evidence.append(f"{source}: {line.strip()}")
    return evidence


def _tail_evidence(stdout: str, stderr: str) -> list[str]:
    lines: list[str] = []
    for source, text in (("stderr", stderr), ("stdout", stdout)):
        tail = [line.strip() for line in text.splitlines() if line.strip()][-4:]
        lines.extend(f"{source}: {line}" for line in tail)
    return lines[-8:]
