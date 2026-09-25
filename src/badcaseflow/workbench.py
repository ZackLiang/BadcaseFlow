from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io_utils import read_json, write_json
from .jobs import list_jobs
from .registry import load_registry
from .rounds import list_rounds


# 工作台是静态导出层，读取已有 run 和 registry，不参与训练任务执行。
DEFAULT_WORKBENCH_OUTPUT = Path("runs/workbench/index.html")


def build_workbench(
    *,
    root: Path,
    output: Path = DEFAULT_WORKBENCH_OUTPUT,
    registry_root: Path | None = None,
    rounds_root: Path | None = None,
    train_runs_root: Path | None = None,
    eval_runs_root: Path | None = None,
    jobs_root: Path | None = None,
    title: str = "BadcaseFlow Workbench",
) -> dict[str, Any]:
    data = collect_workbench_data(
        root=root,
        registry_root=registry_root,
        rounds_root=rounds_root,
        train_runs_root=train_runs_root,
        eval_runs_root=eval_runs_root,
        jobs_root=jobs_root,
        title=title,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output.with_name("workbench.json"), data)
    output.write_text(render_workbench_html(data), encoding="utf-8", newline="\n")
    return {
        "output": str(output),
        "data_output": str(output.with_name("workbench.json")),
        "summary": data["summary"],
    }


def collect_workbench_data(
    *,
    root: Path,
    registry_root: Path | None = None,
    rounds_root: Path | None = None,
    train_runs_root: Path | None = None,
    eval_runs_root: Path | None = None,
    jobs_root: Path | None = None,
    title: str = "BadcaseFlow Workbench",
) -> dict[str, Any]:
    root = root.resolve()
    resolved_registry_root = (registry_root or root).resolve()
    resolved_rounds_root = (rounds_root or (root / "runs" / "rounds")).resolve()
    resolved_train_runs_root = (train_runs_root or (root / "runs" / "train_runs")).resolve()
    resolved_eval_runs_root = (eval_runs_root or (root / "runs" / "eval_runs")).resolve()
    resolved_jobs_root = (jobs_root or (root / "runs" / "jobs")).resolve()

    rounds = list_rounds(resolved_rounds_root)
    registry = load_registry(resolved_registry_root)
    train_runs = _list_run_statuses(resolved_train_runs_root)
    eval_runs = _list_run_statuses(resolved_eval_runs_root)
    jobs = list_jobs(resolved_jobs_root)
    action_rows = _list_iteration_actions(rounds, resolved_rounds_root)
    summary = _summary(rounds, registry, train_runs, eval_runs, jobs, action_rows)

    return {
        "title": title,
        "generated_at": _now(),
        "root": str(root),
        "registry_root": str(resolved_registry_root),
        "rounds_root": str(resolved_rounds_root),
        "train_runs_root": str(resolved_train_runs_root),
        "eval_runs_root": str(resolved_eval_runs_root),
        "jobs_root": str(resolved_jobs_root),
        "summary": summary,
        "rounds": rounds,
        "registry": {
            "models": list(registry.get("models", [])),
            "datasets": list(registry.get("datasets", [])),
            "evaluations": list(registry.get("evaluations", [])),
            "promotions": list(registry.get("promotions", [])),
            "deployments": list(registry.get("deployments", [])),
            "artifacts": list(registry.get("artifacts", [])),
        },
        "train_runs": train_runs,
        "eval_runs": eval_runs,
        "jobs": jobs,
        "actions": action_rows,
    }


def render_workbench_html(data: dict[str, Any]) -> str:
    title = str(data.get("title") or "BadcaseFlow Workbench")
    summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
    registry = data.get("registry") if isinstance(data.get("registry"), dict) else {}
    rounds = _as_rows(data.get("rounds"))
    train_runs = _as_rows(data.get("train_runs"))
    eval_runs = _as_rows(data.get("eval_runs"))
    jobs = _as_rows(data.get("jobs"))
    actions = _as_rows(data.get("actions"))
    models = _as_rows(registry.get("models"))
    datasets = _as_rows(registry.get("datasets"))
    promotions = _as_rows(registry.get("promotions"))
    deployments = _as_rows(registry.get("deployments"))

    lines = [
        "<!doctype html>",
        '<html lang="zh-CN">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{_escape(title)}</title>",
        "<style>",
        _CSS,
        "</style>",
        "</head>",
        "<body>",
        "<header>",
        f"<h1>{_escape(title)}</h1>",
        "<p>Agent 训练飞轮的本地静态工作台，用于查看 round、case action、模型、评测、数据和部署状态。</p>",
        "</header>",
        "<main>",
        '<section class="grid cards">',
        _metric_card("Rounds", summary.get("rounds"), "飞轮轮次"),
        _metric_card("Models", summary.get("models"), "候选模型"),
        _metric_card("Pass Rate", summary.get("latest_pass_rate"), "最新通过率"),
        _metric_card("Actions", summary.get("actions"), "待处理行动"),
        _metric_card("Jobs", summary.get("jobs"), "执行任务"),
        _metric_card("Datasets", summary.get("datasets"), "数据版本"),
        _metric_card("Deployments", summary.get("deployments"), "部署计划"),
        "</section>",
        _section(
            "飞轮轮次",
            "每轮的质量状态、评测结果、候选模型和更新时间。",
            _table(
                rounds,
                [
                    ("round_id", "Round"),
                    ("quality_status", "Quality"),
                    ("workspace_id", "Workspace"),
                    ("pass_rate", "Pass Rate", "summary"),
                    ("failed", "Failed", "summary"),
                    ("model_id", "Model", "summary"),
                    ("promotion_decision", "Decision", "summary"),
                    ("updated_at", "Updated"),
                ],
            ),
        ),
        _section(
            "下一轮行动",
            "从 case report 和 promotion gate 生成的 proposed actions。",
            _table(
                actions,
                [
                    ("round_id", "Round"),
                    ("action_id", "Action"),
                    ("owner", "Owner"),
                    ("priority", "Priority"),
                    ("action_type", "Type"),
                    ("title", "Title"),
                    ("count", "Count"),
                    ("status", "Status"),
                ],
            ),
        ),
        _section(
            "模型注册表",
            "候选模型、训练来源、评测门禁和部署状态。",
            _table(
                models,
                [
                    ("model_id", "Model"),
                    ("status", "Status"),
                    ("backend", "Backend"),
                    ("stage", "Stage"),
                    ("train_run_id", "Train Run"),
                    ("latest_promotion_id", "Promotion"),
                    ("latest_deployment_id", "Deployment"),
                    ("updated_at", "Updated"),
                ],
            ),
        ),
        _section(
            "评测与门禁",
            "模型评测绑定和 promotion decision。",
            _table(
                promotions,
                [
                    ("promotion_id", "Promotion"),
                    ("model_id", "Model"),
                    ("decision", "Decision"),
                    ("pass_rate", "Pass Rate"),
                    ("failed", "Failed"),
                    ("reasons", "Reasons"),
                    ("updated_at", "Updated"),
                ],
            ),
        ),
        _section(
            "执行 Jobs",
            "由 launch plan 提交的本地或 SSH 执行任务。",
            _table(
                jobs,
                [
                    ("job_id", "Job"),
                    ("status", "Status"),
                    ("launcher", "Launcher"),
                    ("kind", "Kind"),
                    ("mode", "Mode"),
                    ("commands_completed", "Done"),
                    ("commands_failed", "Failed"),
                    ("updated_at", "Updated"),
                ],
            ),
        ),
        _section(
            "训练 Runs",
            "顶层训练 harness 运行状态。round 内嵌训练 run 可从飞轮轮次进入查看。",
            _table(
                train_runs,
                [
                    ("train_run_id", "Train Run"),
                    ("status", "Status"),
                    ("backend", "Backend"),
                    ("stage", "Stage"),
                    ("recipe_id", "Recipe"),
                    ("updated_at", "Updated"),
                ],
            ),
        ),
        _section(
            "评测 Runs",
            "顶层评测 harness 运行状态。round 内嵌候选评测 run 可从飞轮轮次进入查看。",
            _table(
                eval_runs,
                [
                    ("eval_run_id", "Eval Run"),
                    ("status", "Status"),
                    ("quality_status", "Quality"),
                    ("backend", "Backend"),
                    ("stage", "Stage"),
                    ("recipe_id", "Recipe"),
                    ("updated_at", "Updated"),
                ],
            ),
        ),
        _section(
            "数据版本",
            "本地 registry 中登记的数据版本和 schema 状态。",
            _table(
                datasets[:80],
                [
                    ("dataset_version_id", "Dataset Version"),
                    ("dataset_id", "Dataset"),
                    ("kind", "Kind"),
                    ("record_count", "Records"),
                    ("schema_status", "Schema"),
                    ("path", "Path"),
                ],
            ),
        ),
        _section(
            "部署计划",
            "已登记的服务计划、灰度比例和入口信息。",
            _table(
                deployments,
                [
                    ("deployment_id", "Deployment"),
                    ("model_id", "Model"),
                    ("backend", "Backend"),
                    ("status", "Status"),
                    ("percent", "Canary %", "canary"),
                    ("endpoint", "Endpoint", "service"),
                    ("updated_at", "Updated"),
                ],
            ),
        ),
        "</main>",
        "<footer>",
        f"Generated at {_escape(data.get('generated_at'))}. Data file: workbench.json",
        "</footer>",
        "</body>",
        "</html>",
    ]
    return "\n".join(lines) + "\n"


def _list_run_statuses(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        status_path = child / "status.json"
        manifest_path = child / "manifest.json"
        if status_path.exists():
            rows.append(read_json(status_path))
        elif manifest_path.exists():
            rows.append(read_json(manifest_path))
    return sorted(rows, key=lambda item: str(item.get("updated_at", "")), reverse=True)


def _list_iteration_actions(rounds: list[dict[str, Any]], rounds_root: Path) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for round_manifest in rounds:
        round_id = str(round_manifest.get("round_id") or "")
        if not round_id:
            continue
        plan_path = rounds_root / round_id / "iteration_plan.json"
        if not plan_path.exists():
            plan_path = _artifact_plan_path(round_manifest, rounds_root)
        if not plan_path or not plan_path.exists():
            continue
        plan = read_json(plan_path)
        for action in _as_rows(plan.get("actions")):
            actions.append(
                {
                    **action,
                    "round_id": round_id,
                    "round_quality_status": round_manifest.get("quality_status"),
                }
            )
    return actions


def _artifact_plan_path(round_manifest: dict[str, Any], rounds_root: Path) -> Path | None:
    artifacts = round_manifest.get("artifacts") if isinstance(round_manifest.get("artifacts"), dict) else {}
    relative = artifacts.get("iteration_plan")
    round_id = str(round_manifest.get("round_id") or "")
    if not relative or not round_id:
        return None
    return rounds_root / round_id / str(relative)


def _summary(
    rounds: list[dict[str, Any]],
    registry: dict[str, Any],
    train_runs: list[dict[str, Any]],
    eval_runs: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
    actions: list[dict[str, Any]],
) -> dict[str, Any]:
    latest = rounds[0] if rounds else {}
    latest_summary = latest.get("summary") if isinstance(latest.get("summary"), dict) else {}
    return {
        "rounds": len(rounds),
        "blocked_rounds": sum(1 for item in rounds if item.get("quality_status") == "blocked"),
        "approved_rounds": sum(1 for item in rounds if item.get("quality_status") == "approved"),
        "models": len(registry.get("models", [])),
        "datasets": len(registry.get("datasets", [])),
        "evaluations": len(registry.get("evaluations", [])),
        "promotions": len(registry.get("promotions", [])),
        "deployments": len(registry.get("deployments", [])),
        "train_runs": len(train_runs),
        "eval_runs": len(eval_runs),
        "jobs": len(jobs),
        "failed_jobs": sum(1 for item in jobs if item.get("status") in {"failed", "timed_out"}),
        "actions": len(actions),
        "latest_round_id": latest.get("round_id"),
        "latest_quality_status": latest.get("quality_status"),
        "latest_pass_rate": latest_summary.get("pass_rate"),
        "latest_failed": latest_summary.get("failed"),
    }


def _section(title: str, description: str, body: str) -> str:
    return "\n".join(
        [
            '<section class="panel">',
            f"<h2>{_escape(title)}</h2>",
            f"<p>{_escape(description)}</p>",
            body,
            "</section>",
        ]
    )


def _metric_card(label: str, value: Any, hint: str) -> str:
    return "\n".join(
        [
            '<article class="card">',
            f"<span>{_escape(hint)}</span>",
            f"<strong>{_escape(_cell(value))}</strong>",
            f"<em>{_escape(label)}</em>",
            "</article>",
        ]
    )


def _table(rows: list[dict[str, Any]], columns: list[tuple[str, str] | tuple[str, str, str]]) -> str:
    if not rows:
        return '<div class="empty">暂无记录</div>'
    header = "".join(f"<th>{_escape(label)}</th>" for _, label, *_ in columns)
    body_lines = []
    for row in rows:
        cells = []
        for column in columns:
            key = column[0]
            nested = column[2] if len(column) > 2 else None
            value = _nested_value(row, key, nested)
            cells.append(f"<td>{_escape(_cell(value))}</td>")
        body_lines.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<div class="table-wrap"><table>'
        f"<thead><tr>{header}</tr></thead>"
        "<tbody>"
        + "".join(body_lines)
        + "</tbody></table></div>"
    )


def _nested_value(row: dict[str, Any], key: str, nested: str | None) -> Any:
    if not nested:
        return row.get(key)
    parent = row.get(nested)
    if isinstance(parent, dict):
        return parent.get(key)
    return None


def _as_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _cell(value: Any) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else "-"
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _escape(value: Any) -> str:
    return html.escape(_cell(value), quote=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


_CSS = """
:root {
  color-scheme: light;
  --bg: #f6f7f9;
  --panel: #ffffff;
  --text: #20242a;
  --muted: #667085;
  --line: #d9dee7;
  --accent: #0f766e;
  --accent-2: #2563eb;
}
* {
  box-sizing: border-box;
}
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  line-height: 1.55;
}
header {
  padding: 32px 40px 24px;
  background: #102033;
  color: #fff;
}
h1 {
  margin: 0 0 8px;
  font-size: 32px;
  font-weight: 720;
}
header p,
.panel p {
  margin: 0;
  color: var(--muted);
}
header p {
  color: #d8dee9;
}
main {
  max-width: 1280px;
  margin: 0 auto;
  padding: 24px;
}
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 12px;
}
.card,
.panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
}
.card {
  padding: 16px;
}
.card span,
.card em {
  display: block;
  color: var(--muted);
  font-size: 13px;
  font-style: normal;
}
.card strong {
  display: block;
  margin: 8px 0 2px;
  font-size: 28px;
}
.panel {
  margin-top: 18px;
  padding: 18px;
}
h2 {
  margin: 0 0 6px;
  font-size: 20px;
}
.table-wrap {
  overflow-x: auto;
  margin-top: 14px;
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
}
th,
td {
  padding: 10px 12px;
  border-bottom: 1px solid var(--line);
  text-align: left;
  vertical-align: top;
  white-space: nowrap;
}
td:last-child {
  white-space: normal;
}
th {
  color: #344054;
  font-weight: 650;
  background: #f8fafc;
}
.empty {
  margin-top: 14px;
  padding: 16px;
  border: 1px dashed var(--line);
  border-radius: 8px;
  color: var(--muted);
}
footer {
  max-width: 1280px;
  margin: 0 auto;
  padding: 0 24px 32px;
  color: var(--muted);
  font-size: 13px;
}
@media (max-width: 720px) {
  header {
    padding: 24px 20px 18px;
  }
  main {
    padding: 16px;
  }
  .panel {
    padding: 14px;
  }
}
""".strip()
