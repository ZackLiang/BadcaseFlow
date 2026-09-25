from __future__ import annotations

import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io_utils import read_json, write_json
from .recipes import build_train_dry_run


# remote 配置只记录连接和目录信息，真正的执行仍由生成的计划或 job 负责。
def add_remote(
    config_path: Path,
    name: str,
    host: str,
    workdir: str,
    port: int = 22,
    python: str = "python",
) -> dict[str, Any]:
    config = _load_remote_config(config_path)
    config.setdefault("remotes", {})[name] = {
        "host": host,
        "port": port,
        "workdir": workdir.rstrip("/"),
        "python": python,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(config_path, config)
    return config["remotes"][name]


def list_remotes(config_path: Path) -> dict[str, Any]:
    return _load_remote_config(config_path).get("remotes", {})


def build_remote_plan(
    config_path: Path,
    target: str,
    run_dir: Path,
    recipe_path: Path,
    workspace: str | None = None,
) -> dict[str, Any]:
    remotes = list_remotes(config_path)
    if target not in remotes:
        raise KeyError(f"remote target not found: {target}")
    remote = remotes[target]
    run_dir = run_dir.resolve()
    recipe_path = recipe_path.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run directory not found: {run_dir}")
    if not recipe_path.exists():
        raise FileNotFoundError(f"recipe not found: {recipe_path}")

    train_manifest = build_train_dry_run(recipe_path, workspace=workspace)
    remote_root = remote["workdir"]
    remote_run_dir = f"{remote_root}/runs/{run_dir.name}"
    remote_recipe_dir = f"{remote_root}/recipes"
    remote_recipe_path = f"{remote_recipe_dir}/{recipe_path.name}"
    port = int(remote.get("port", 22))
    host = remote["host"]
    python = remote.get("python", "python")

    commands = {
        "prepare": [
            f"ssh -p {port} {host} {shlex.quote(f'mkdir -p {remote_run_dir} {remote_recipe_dir}')}",
        ],
        "sync": [
            f"scp -P {port} -r {shlex.quote(str(run_dir))} {host}:{shlex.quote(remote_run_dir.rsplit('/', 1)[0])}/",
            f"scp -P {port} {shlex.quote(str(recipe_path))} {host}:{shlex.quote(remote_recipe_path)}",
        ],
        "train_dry_run": [
            "ssh -p {port} {host} {cmd}".format(
                port=port,
                host=host,
                cmd=shlex.quote(
                    f"cd {remote_root} && {python} -m badcaseflow.cli train dry-run "
                    f"--workspace {workspace or ''} --recipe {remote_recipe_path} --output-root runs"
                ),
            )
        ],
        "collect": [
            f"scp -P {port} -r {host}:{shlex.quote(remote_run_dir)} {shlex.quote(str(run_dir.parent))}",
        ],
    }
    return {
        "target": target,
        "remote": remote,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "local": {"run_dir": str(run_dir), "recipe": str(recipe_path)},
        "remote_paths": {
            "root": remote_root,
            "run_dir": remote_run_dir,
            "recipe": remote_recipe_path,
        },
        "commands": commands,
        "train_manifest": train_manifest,
    }


def render_plan_script(plan: dict[str, Any]) -> str:
    lines = [
        "# BadcaseFlow remote dry-run plan",
        "$ErrorActionPreference = 'Stop'",
        "",
    ]
    for group in ("prepare", "sync", "train_dry_run", "collect"):
        lines.append(f"# {group}")
        for command in plan["commands"].get(group, []):
            lines.append(command)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _load_remote_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {"remotes": {}}
    return read_json(config_path)
