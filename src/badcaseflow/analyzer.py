from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any


ACTION_BY_FAILURE = {
    "tool_missing": "ADD_SFT_DATA",
    "tool_observation_insufficient": "PATCH_TOOL_SCHEMA",
    "unsafe_side_effect": "PATCH_WORKFLOW",
    "citation_missing": "ADD_SFT_DATA",
    "answer_not_supported": "ADD_EVAL_CASE",
    "safe_escalation_missing": "ADD_PREFERENCE_PAIR",
    "wrong_answer": "ADD_SFT_DATA",
    "reasoning_trace_missing": "ADD_SFT_DATA",
    "missing_trace": "RERUN_ROLLOUT",
}


ROOT_CAUSE_BY_FAILURE = {
    "tool_missing": "Agent did not use the required tool.",
    "tool_observation_insufficient": "Tool result did not satisfy task requirements.",
    "unsafe_side_effect": "Agent performed an action that should require confirmation.",
    "citation_missing": "Answer did not cite retrieved evidence.",
    "answer_not_supported": "Answer was not grounded in the observation.",
    "safe_escalation_missing": "Agent did not escalate or ask for approval when required.",
    "wrong_answer": "Final answer does not match the expected result.",
    "reasoning_trace_missing": "Final answer lacks calculation or reasoning details.",
    "missing_trace": "Evaluation could not find a trace for the sample.",
}


def analyze_cases(
    eval_results: list[dict[str, Any]], traces: list[dict[str, Any]]
) -> dict[str, Any]:
    trace_by_id = {trace["trace_id"]: trace for trace in traces}
    cases: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()
    groups: dict[str, list[str]] = defaultdict(list)

    for result in eval_results:
        if result.get("passed"):
            continue
        failure_types = result.get("failure_types") or ["eval_failed"]
        primary = failure_types[0]
        action = ACTION_BY_FAILURE.get(primary, "NEEDS_REVIEW")
        trace = trace_by_id.get(result.get("trace_id"))
        case = {
            "case_id": f"case-{result['sample_id']}",
            "trace_id": result.get("trace_id"),
            "sample_id": result["sample_id"],
            "primary_failure_type": primary,
            "failure_types": failure_types,
            "root_cause": ROOT_CAUSE_BY_FAILURE.get(primary, "Needs manual review."),
            "recommended_action": action,
            "target_dataset": _target_dataset(action),
            "needs_human_review": action in {"ADD_PREFERENCE_PAIR", "PATCH_WORKFLOW", "NEEDS_REVIEW"},
            "reasons": result.get("reasons", []),
            "final_answer": trace.get("final_answer") if trace else None,
        }
        cases.append(case)
        action_counts[action] += 1
        for failure_type in failure_types:
            failure_counts[failure_type] += 1
            groups[failure_type].append(case["case_id"])

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_failed_cases": len(cases),
        "failure_counts": dict(sorted(failure_counts.items())),
        "recommended_action_counts": dict(sorted(action_counts.items())),
        "groups": dict(sorted(groups.items())),
        "cases": cases,
    }


def render_case_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# BadcaseFlow Case Report",
        "",
        f"- Generated at: `{report.get('generated_at')}`",
        f"- Failed cases: `{report.get('total_failed_cases', 0)}`",
        "",
        "## Failure Types",
        "",
    ]
    failure_counts = report.get("failure_counts", {})
    if failure_counts:
        lines.extend(["| Failure Type | Count |", "|---|---:|"])
        for failure_type, count in failure_counts.items():
            lines.append(f"| `{failure_type}` | {count} |")
    else:
        lines.append("No failures.")

    lines.extend(["", "## Recommended Actions", ""])
    action_counts = report.get("recommended_action_counts", {})
    if action_counts:
        lines.extend(["| Action | Count |", "|---|---:|"])
        for action, count in action_counts.items():
            lines.append(f"| `{action}` | {count} |")
    else:
        lines.append("No actions required.")

    lines.extend(["", "## Cases", ""])
    for case in report.get("cases", []):
        lines.extend(
            [
                f"### {case.get('case_id')}",
                "",
                f"- Sample: `{case.get('sample_id')}`",
                f"- Trace: `{case.get('trace_id')}`",
                f"- Failure: `{case.get('primary_failure_type')}`",
                f"- Root cause: {case.get('root_cause')}",
                f"- Recommended action: `{case.get('recommended_action')}`",
                f"- Needs human review: `{case.get('needs_human_review')}`",
                "",
            ]
        )
        reasons = case.get("reasons", [])
        if reasons:
            lines.append("Reasons:")
            for reason in reasons:
                lines.append(f"- {reason}")
            lines.append("")
        final_answer = case.get("final_answer")
        if final_answer:
            lines.extend(["Final answer:", "", "```text", final_answer, "```", ""])
    return "\n".join(lines).rstrip() + "\n"


def _target_dataset(action: str) -> str | None:
    if action == "ADD_SFT_DATA":
        return "sft_candidate"
    if action == "ADD_PREFERENCE_PAIR":
        return "preference_candidate"
    if action == "ADD_EVAL_CASE":
        return "regression_eval"
    if action == "RERUN_ROLLOUT":
        return "rollout_retry"
    return None
