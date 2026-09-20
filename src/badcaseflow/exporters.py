from __future__ import annotations

from typing import Any


SYSTEM_PROMPT = (
    "You are a task-oriented agent. Use tools when needed, ground answers in observations, "
    "and ask for confirmation before irreversible actions."
)


def export_sft(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for trace in traces:
        records.append(
            {
                "id": trace["trace_id"],
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": trace.get("input", {}).get("query", "")},
                    {"role": "assistant", "content": trace.get("final_answer", "")},
                ],
                "metadata": {
                    "sample_id": trace["sample_id"],
                    "source_trace_id": trace["trace_id"],
                    "tool_bundle_id": trace.get("tool_bundle_id"),
                    "dataset_role": "sft",
                },
            }
        )
    return records


def export_preference(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for case in cases:
        rejected = case.get("final_answer") or ""
        chosen = _chosen_answer_for_case(case)
        records.append(
            {
                "id": f"pref-{case['case_id']}",
                "prompt": case.get("sample_id", ""),
                "chosen": chosen,
                "rejected": rejected,
                "metadata": {
                    "case_id": case["case_id"],
                    "trace_id": case.get("trace_id"),
                    "primary_failure_type": case["primary_failure_type"],
                    "needs_review": True,
                },
            }
        )
    return records


def export_rl_prompts(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for task in tasks:
        records.append(
            {
                "id": f"rl-{task['sample_id']}",
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": task.get("input", {}).get("query", "")},
                ],
                "reward_model": {
                    "style": "rule",
                    "success_criteria": task.get("expected", {}).get("success_criteria", []),
                },
                "metadata": {"sample_id": task["sample_id"], "tags": task.get("tags", [])},
            }
        )
    return records


def _chosen_answer_for_case(case: dict[str, Any]) -> str:
    failure = case["primary_failure_type"]
    if failure == "unsafe_side_effect":
        return "I found available options, but I will ask for confirmation before booking or reserving anything."
    if failure == "citation_missing":
        return "The answer should cite the retrieved evidence and avoid unsupported policy claims."
    if failure == "wrong_answer":
        return "The answer should use the calculator result and present the exact amount with the formula."
    return "The answer should complete the task while following tool, evidence, and safety requirements."

