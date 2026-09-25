from __future__ import annotations

import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .remote import list_remotes


LAUNCH_KINDS = {"train", "eval"}
LAUNCH_MODES = {"dry-run", "plan", "run"}
LAUNCHERS = {"local", "ssh"}


# launcher 只生成可审查的执行计划；真正执行由 job runner 或用户确认后的脚本完成。
def build_launch_plan(
    *,
    kind: str,
    recipe_path: Path,
    workspace: str | None = None,
    launcher: str = "local",
    mode: str = "dry-run",
    output_root: Path = Path("runs"),
    run_id: str | None = None,
    config_path: Path | None = None,
    target: str | None = None,
    python: str = "python",
    cwd: Path | None = None,
    includes: list[Path] | None = None,
) -> dict[str, Any]:
    kind = kind.strip().lower()
    launcher = launcher.strip().lower()
    mode = mode.strip().lower()
    if kind not in LAUNCH_KINDS:
        raise ValueError(f"kind must be one of: {', '.join(sorted(LAUNCH_KINDS))}")
    if launcher not in LAUNCHERS:
        raise ValueError(f"launcher must be one of: {', '.join(sorted(LAUNCHERS))}")
    if mode not in LAUNCH_MODES:
        raise ValueError(f"mode must be one of: {', '.join(sorted(LAUNCH_MODES))}")
    if not recipe_path.exists():
        raise FileNotFoundError(f"recipe not found: {recipe_path}")

    if launcher == "ssh":
        if config_path is None:
            raise ValueError("config_path is required for ssh launcher")
        if not target:
            raise ValueError("target is required for ssh launcher")
        return _build_ssh_plan(
            kind=kind,
            recipe_path=recipe_path,
            workspace=workspace,
            mode=mode,
            output_root=output_root,
            run_id=run_id,
            config_path=config_path,
            target=target,
            python=python,
            includes=includes or [],
        )
    return _build_local_plan(
        kind=kind,
        recipe_path=recipe_path,
        workspace=workspace,
        mode=mode,
        output_root=output_root,
        run_id=run_id,
        python=python,
        cwd=cwd,
    )


def render_launch_script(plan: dict[str, Any]) -> str:
    lines = [
        "# BadcaseFlow launch plan",
        "$ErrorActionPreference = 'Stop'",
        "",
    ]
    for group in ("prepare", "sync", "run", "collect"):
        commands = plan.get("commands", {}).get(group, [])
        if not commands:
            continue
        lines.append(f"# {group}")
        lines.extend(str(command) for command in commands)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _build_local_plan(
    *,
    kind: str,
    recipe_path: Path,
    workspace: str | None,
    mode: str,
    output_root: Path,
    run_id: str | None,
    python: str,
    cwd: Path | None,
) -> dict[str, Any]:
    command = _bcf_command(
        kind=kind,
        recipe_path=recipe_path,
        workspace=workspace,
        mode=mode,
        output_root=output_root,
        run_id=run_id,
        python=python,
    )
    return {
        "plan_id": _plan_id(kind, "local", run_id),
        "kind": kind,
        "launcher": "local",
        "mode": mode,
        "created_at": _now(),
        "cwd": str((cwd or Path(".")).resolve()),
        "recipe": str(recipe_path),
        "output_root": str(output_root),
        "run_id": run_id,
        "commands": {"prepare": [], "sync": [], "run": [_local_command(command)], "collect": []},
    }


def _build_ssh_plan(
    *,
    kind: str,
    recipe_path: Path,
    workspace: str | None,
    mode: str,
    output_root: Path,
    run_id: str | None,
    config_path: Path,
    target: str,
    python: str,
    includes: list[Path],
) -> dict[str, Any]:
    remotes = list_remotes(config_path)
    if target not in remotes:
        raise KeyError(f"remote target not found: {target}")
    remote = remotes[target]
    host = remote["host"]
    port = int(remote.get("port", 22))
    remote_python = remote.get("python") or python
    remote_root = str(remote["workdir"]).rstrip("/")
    remote_recipe_dir = f"{remote_root}/recipes"
    remote_recipe_path = f"{remote_recipe_dir}/{recipe_path.name}"
    remote_output_root = "runs"
    remote_include_dir = f"{remote_root}/inputs"

    remote_command = _bcf_command(
        kind=kind,
        recipe_path=remote_recipe_path,
        workspace=workspace,
        mode=mode,
        output_root=remote_output_root,
        run_id=run_id,
        python=remote_python,
        remote=True,
    )
    prepare = [
        _ssh(host, port, f"mkdir -p {shlex.quote(remote_recipe_dir)} {shlex.quote(remote_include_dir)} {shlex.quote(f'{remote_root}/runs')}")
    ]
    sync = [
        _scp_to(host, port, recipe_path, remote_recipe_path),
    ]
    for item in includes:
        sync.append(_scp_to(host, port, item, f"{remote_include_dir}/{item.name}"))
    collect = []
    remote_run_path = _remote_run_path(remote_root, kind, mode, run_id)
    if remote_run_path:
        collect.append(_scp_from(host, port, remote_run_path, output_root))

    return {
        "plan_id": _plan_id(kind, f"ssh-{target}", run_id),
        "kind": kind,
        "launcher": "ssh",
        "mode": mode,
        "target": target,
        "created_at": _now(),
        "recipe": str(recipe_path),
        "output_root": str(output_root),
        "run_id": run_id,
        "remote": {
            "host": host,
            "port": port,
            "workdir": remote_root,
            "recipe": remote_recipe_path,
            "output_root": f"{remote_root}/{remote_output_root}",
        },
        "commands": {
            "prepare": prepare,
            "sync": sync,
            "run": [_ssh(host, port, f"cd {shlex.quote(remote_root)} && {_shell_join(remote_command)}")],
            "collect": collect,
        },
    }


def _bcf_command(
    *,
    kind: str,
    recipe_path: Path | str,
    workspace: str | None,
    mode: str,
    output_root: Path | str,
    run_id: str | None,
    python: str,
    remote: bool = False,
) -> list[str]:
    command = [python, "-m", "badcaseflow.cli"]
    if kind == "train":
        command.extend(["train", "dry-run" if mode == "dry-run" else "run"])
    else:
        command.extend(["evals", "dry-run" if mode == "dry-run" else "run"])
    if workspace:
        command.extend(["--workspace", workspace])
    command.extend(["--recipe", str(recipe_path), "--output-root", str(output_root)])
    if mode in {"plan", "run"} and run_id:
        command.extend(["--run-id", run_id])
    if mode == "plan":
        command.append("--plan-only")
    return command


def _remote_run_path(remote_root: str, kind: str, mode: str, run_id: str | None) -> str | None:
    if mode == "dry-run":
        return f"{remote_root}/runs/dry_runs"
    if not run_id:
        return None
    subdir = "train_runs" if kind == "train" else "eval_runs"
    return f"{remote_root}/runs/{subdir}/{run_id}"


def _ssh(host: str, port: int, command: str) -> str:
    return f"ssh -p {port} {host} {shlex.quote(command)}"


def _scp_to(host: str, port: int, local_path: Path, remote_path: str) -> str:
    flag = "-r " if local_path.is_dir() else ""
    return f"scp -P {port} {flag}{_quote_local(local_path)} {host}:{shlex.quote(remote_path)}"


def _scp_from(host: str, port: int, remote_path: str, local_dir: Path) -> str:
    return f"scp -P {port} -r {host}:{shlex.quote(remote_path)} {_quote_local(local_dir)}"


def _local_command(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def _shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def _quote_local(path: Path) -> str:
    return shlex.quote(str(path))


def _plan_id(kind: str, launcher: str, run_id: str | None) -> str:
    suffix = run_id or datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"launch-{kind}-{launcher}-{suffix}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
