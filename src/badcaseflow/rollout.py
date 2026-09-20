from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def rollout_tasks(tasks: list[dict[str, Any]], agent: str = "mock") -> list[dict[str, Any]]:
    if agent not in {"mock", "mock-buggy"}:
        raise ValueError("only mock and mock-buggy agents are available in v0.1")
    return [_rollout_one(task, agent=agent) for task in tasks]


def _rollout_one(task: dict[str, Any], agent: str) -> dict[str, Any]:
    sample_id = task["sample_id"]
    query = task["input"]["query"]
    tags = set(task.get("tags", []))
    trace_id = f"trace-{sample_id}"
    steps: list[dict[str, Any]]

    if "calendar" in tags:
        final_answer = (
            "I found two available 60 minute slots tomorrow: 14:00 and 15:00. "
            "I booked the 14:00 slot for the project review."
        )
        if agent == "mock-buggy":
            final_answer = "I could not find any calendar slots."
        steps = [
            _tool_step(
                "calendar.search_slots",
                {"date": "tomorrow", "duration_minutes": 60},
                {"slots": ["14:00", "15:00"]},
            )
        ]
    elif "calculator" in tags:
        expression = "(19.99 + 42.50 + 8.25) * 0.9"
        steps = [_tool_step("calculator.evaluate", {"expression": expression}, {"result": 63.67})]
        final_answer = "Refund amount: 63.67 USD. Formula: (19.99 + 42.50 + 8.25) * 0.9."
    elif "rag" in tags or "policy" in tags:
        steps = [
            _tool_step(
                "rag.search_policy",
                {"query": query, "top_k": 3},
                {
                    "articles": [
                        {
                            "article_id": "POL-1",
                            "text": "Approval is required when a request has uncertain evidence or crosses a configured review threshold.",
                        }
                    ]
                },
            )
        ]
        final_answer = (
            "Approval is required if the request crosses the review threshold or the evidence is uncertain. "
            "Evidence: [POL-1]."
        )
    else:
        steps = []
        final_answer = "I do not know which tool should be used for this task."

    return {
        "trace_id": trace_id,
        "sample_id": sample_id,
        "workspace_id": task.get("workspace_id"),
        "agent": agent,
        "model_id": "mock-agent@v0.1",
        "prompt_version": "mock-prompt@v0.1",
        "tool_bundle_id": "demo-tools@v0.1",
        "input": task.get("input", {}),
        "steps": steps,
        "final_answer": final_answer,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lineage": {"source_sample_id": sample_id},
    }


def _tool_step(tool_name: str, arguments: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "tool_call",
        "tool_name": tool_name,
        "arguments": arguments,
        "validation": {"schema": "passed", "policy": "passed"},
        "observation": observation,
    }

