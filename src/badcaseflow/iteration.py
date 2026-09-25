from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io_utils import read_json, write_json


# iteration plan 是 case report 到下一轮工程动作之间的稳定接口。
ACTION_TEMPLATES: dict[str, dict[str, Any]] = {
    "PATCH_WORKFLOW": {
        "title": "修复 Agent 工作流",
        "priority": "P0",
        "rationale": "失败样本涉及流程控制或确认步骤，优先改工作流，再考虑补训练数据。",
        "suggested_outputs": ["workflow_patch", "regression_eval_case"],
    },
    "PATCH_TOOL_SCHEMA": {
        "title": "修复工具 schema 或工具返回协议",
        "priority": "P0",
        "rationale": "工具调用或 observation 协议不足会污染训练数据，需要先修接口契约。",
        "suggested_outputs": ["tool_schema_patch", "tool_contract_test", "regression_eval_case"],
    },
    "ADD_EVAL_CASE": {
        "title": "加入回归评测样本",
        "priority": "P1",
        "rationale": "评测暴露了未覆盖的失败模式，应先冻结成回归用例。",
        "suggested_outputs": ["regression_eval_case"],
    },
    "ADD_PREFERENCE_PAIR": {
        "title": "构造偏好数据",
        "priority": "P1",
        "rationale": "边界行为需要 chosen/rejected 对来表达偏好，而不是只追加正样本。",
        "suggested_outputs": ["preference_candidate"],
    },
    "ADD_SFT_DATA": {
        "title": "补充 SFT 数据",
        "priority": "P2",
        "rationale": "失败更像能力或格式缺口，可以补充高质量示范轨迹。",
        "suggested_outputs": ["sft_candidate"],
    },
    "RERUN_ROLLOUT": {
        "title": "重新生成 rollout",
        "priority": "P1",
        "rationale": "缺失 trace 或 rollout 异常时，先重跑数据生成链路。",
        "suggested_outputs": ["trace_retry"],
    },
    "NEEDS_REVIEW": {
        "title": "人工复核失败样本",
        "priority": "P1",
        "rationale": "失败原因暂不明确，需要先补充标签或备注。",
        "suggested_outputs": ["reviewed_case_labels"],
    },
}


def build_iteration_plan(
    *,
    round_manifest: dict[str, Any],
    eval_report: dict[str, Any],
    case_report: dict[str, Any],
    promotion: dict[str, Any] | None = None,
) -> dict[str, Any]:
    round_id = str(round_manifest.get("round_id") or "round")
    actions = _case_actions(case_report)
    decision = promotion.get("decision") if promotion else None
    failed = int(eval_report.get("failed", 0) or 0)
    pass_rate = float(eval_report.get("pass_rate", 0.0) or 0.0)

    if decision == "blocked":
        actions.insert(
            0,
            _action(
                len(actions) + 1,
                "BLOCK_PROMOTION",
                "暂缓候选模型进入下一阶段",
                "P0",
                "候选模型没有通过本轮门禁，应先处理失败样本或降低不确定性。",
                ["case_fix_plan", "rerun_eval"],
                source="promotion_gate",
                count=failed,
            ),
        )
    elif decision == "approved":
        actions.append(
            _action(
                len(actions) + 1,
                "PROMOTE_CANDIDATE",
                "准备候选模型进入下一阶段",
                "P1",
                "候选模型已通过当前门禁，可以进入灰度、离线对比或发布准备。",
                ["release_candidate_record", "serving_plan"],
                source="promotion_gate",
                count=1,
            )
        )
    elif failed == 0 and pass_rate > 0:
        actions.append(
            _action(
                len(actions) + 1,
                "EXPAND_EVAL",
                "扩展评测覆盖面",
                "P2",
                "当前样本全部通过，下一轮应增加更难样本或新场景，避免只优化现有集合。",
                ["new_eval_slice", "hard_case_seed"],
                source="eval_report",
                count=1,
            )
        )

    return {
        "plan_id": f"plan-{round_id}",
        "round_id": round_id,
        "workspace_id": round_manifest.get("workspace_id"),
        "generated_at": _now(),
        "status": "proposed",
        "quality_status": _quality_status(eval_report, promotion),
        "summary": {
            "total": eval_report.get("total", 0),
            "passed": eval_report.get("passed", 0),
            "failed": failed,
            "pass_rate": pass_rate,
            "promotion_decision": decision,
            "action_count": len(actions),
        },
        "actions": _renumber(actions),
    }


def build_iteration_plan_from_round(round_dir: Path) -> dict[str, Any]:
    manifest = read_json(round_dir / "round.json")
    eval_key = "candidate_eval_report" if _has_artifact(manifest, "candidate_eval_report") else "eval_report"
    case_key = "candidate_case_report" if _has_artifact(manifest, "candidate_case_report") else "case_report"
    eval_report = read_json(round_dir / _artifact_path(manifest, eval_key))
    case_report = read_json(round_dir / _artifact_path(manifest, case_key))
    promotion_decision = (manifest.get("summary") or {}).get("promotion_decision")
    promotion = {"decision": promotion_decision} if promotion_decision else None
    return build_iteration_plan(
        round_manifest=manifest,
        eval_report=eval_report,
        case_report=case_report,
        promotion=promotion,
    )


def write_iteration_plan(round_dir: Path, plan: dict[str, Any], output: Path | None = None) -> dict[str, str]:
    plan_path = output or (round_dir / "iteration_plan.json")
    markdown_path = plan_path.with_suffix(".md")
    write_json(plan_path, plan)
    markdown_path.write_text(render_iteration_plan_markdown(plan), encoding="utf-8", newline="\n")
    return {"iteration_plan": _relative(plan_path, round_dir), "iteration_plan_md": _relative(markdown_path, round_dir)}


def render_iteration_plan_markdown(plan: dict[str, Any]) -> str:
    summary = plan.get("summary") or {}
    lines = [
        "# BadcaseFlow Iteration Plan",
        "",
        f"- Round: `{plan.get('round_id')}`",
        f"- Status: `{plan.get('status')}`",
        f"- Quality: `{plan.get('quality_status')}`",
        f"- Pass rate: `{summary.get('pass_rate')}`",
        f"- Failed: `{summary.get('failed')}`",
        "",
        "## Actions",
        "",
    ]
    actions = plan.get("actions") or []
    if not actions:
        lines.append("No action proposed.")
    else:
        lines.extend(
            [
                "| ID | Status | Owner | Priority | Type | Title | Count |",
                "|---|---|---|---|---|---|---:|",
            ]
        )
        for action in actions:
            lines.append(
                "| {id} | {status} | {owner} | {priority} | `{type}` | {title} | {count} |".format(
                    id=action.get("action_id"),
                    status=action.get("status") or "proposed",
                    owner=action.get("owner") or "-",
                    priority=action.get("priority"),
                    type=action.get("action_type"),
                    title=action.get("title"),
                    count=action.get("count", 0),
                )
            )
    lines.append("")
    return "\n".join(lines)


def _case_actions(case_report: dict[str, Any]) -> list[dict[str, Any]]:
    cases = case_report.get("cases") or []
    by_action: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        action_type = str(case.get("recommended_action") or "NEEDS_REVIEW")
        by_action.setdefault(action_type, []).append(case)

    actions: list[dict[str, Any]] = []
    for action_type, group in sorted(by_action.items(), key=lambda item: _priority_sort(item[0])):
        template = ACTION_TEMPLATES.get(action_type, ACTION_TEMPLATES["NEEDS_REVIEW"])
        target_datasets = sorted({str(case.get("target_dataset")) for case in group if case.get("target_dataset")})
        actions.append(
            _action(
                len(actions) + 1,
                action_type,
                template["title"],
                template["priority"],
                template["rationale"],
                template["suggested_outputs"],
                source="case_report",
                count=len(group),
                case_ids=[str(case.get("case_id")) for case in group],
                target_datasets=target_datasets,
                needs_human_review=any(bool(case.get("needs_human_review")) for case in group),
            )
        )
    return actions


def _action(
    index: int,
    action_type: str,
    title: str,
    priority: str,
    rationale: str,
    suggested_outputs: list[str],
    *,
    source: str,
    count: int,
    case_ids: list[str] | None = None,
    target_datasets: list[str] | None = None,
    needs_human_review: bool = False,
) -> dict[str, Any]:
    return {
        "action_id": f"action-{index:03d}",
        "action_type": action_type,
        "title": title,
        "priority": priority,
        "status": "proposed",
        "source": source,
        "count": count,
        "case_ids": case_ids or [],
        "target_datasets": target_datasets or [],
        "needs_human_review": needs_human_review,
        "rationale": rationale,
        "suggested_outputs": suggested_outputs,
    }


def _renumber(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    priority_order = {"P0": 0, "P1": 1, "P2": 2}
    rows = sorted(actions, key=lambda item: (priority_order.get(str(item.get("priority")), 9), item["action_type"]))
    for index, action in enumerate(rows, start=1):
        action["action_id"] = f"action-{index:03d}"
    return rows


def _priority_sort(action_type: str) -> tuple[int, str]:
    template = ACTION_TEMPLATES.get(action_type, ACTION_TEMPLATES["NEEDS_REVIEW"])
    priority_order = {"P0": 0, "P1": 1, "P2": 2}
    return priority_order.get(str(template.get("priority")), 9), action_type


def _quality_status(eval_report: dict[str, Any], promotion: dict[str, Any] | None) -> str:
    if promotion and promotion.get("decision"):
        return str(promotion["decision"])
    return "passed" if int(eval_report.get("failed", 0) or 0) == 0 else "needs_iteration"


def _artifact_path(manifest: dict[str, Any], key: str) -> Path:
    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}
    value = artifacts.get(key)
    if not value:
        raise FileNotFoundError(f"round artifact not found in manifest: {key}")
    return Path(str(value))


def _has_artifact(manifest: dict[str, Any], key: str) -> bool:
    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}
    return bool(artifacts.get(key))


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
