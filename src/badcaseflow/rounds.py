from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .analyzer import analyze_cases, render_case_report_markdown
from .datasets import build_dataset_version, dataset_summary, write_dataset_collection
from .evaluation import evaluate_traces
from .evals.harness import create_eval_run_id, plan_eval_run, run_eval_manifest
from .exporters import export_preference, export_rl_prompts, export_sft
from .io_utils import read_json, read_jsonl, write_json, write_jsonl
from .iteration import build_iteration_plan, write_iteration_plan
from .recipes import build_eval_dry_run, build_train_dry_run
from .registry import (
    attach_eval_report,
    attach_eval_run,
    index_eval_run,
    index_train_run,
    register_dataset_manifest,
    register_model,
    registry_path,
)
from .rollout import rollout_tasks
from .schemas import validate_task_sample
from .store import LocalRunStore
from .training.harness import create_train_run_id, plan_train_run, run_train_manifest


# 一个 round 把数据生成、评测、case 分析、训练、候选回评测和下一轮 action
# 放进同一份 manifest，便于复盘每次迭代到底改变了什么。
ROUND_FILENAME = "round.json"
ROUND_EVENTS_FILENAME = "events.jsonl"
TRAIN_MODES = {"dry-run", "plan", "run"}
EVAL_MODES = {"dry-run", "plan", "run"}


def create_round_id(workspace_id: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"round-{_slug(workspace_id)}-{timestamp}"


def load_round(round_dir: Path) -> dict[str, Any]:
    return read_json(round_dir / ROUND_FILENAME)


def list_rounds(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        manifest_path = child / ROUND_FILENAME
        if manifest_path.exists():
            rows.append(read_json(manifest_path))
    return sorted(rows, key=lambda item: str(item.get("updated_at", "")), reverse=True)


def run_local_round(
    *,
    workspace_id: str,
    round_dir: Path,
    seed_path: Path,
    suite_path: Path,
    agent: str = "mock",
    recipe_path: Path | None = None,
    train_mode: str = "dry-run",
    registry_root: Path | None = None,
    model_id: str | None = None,
    model_path: str | None = None,
    train_run_id: str | None = None,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout_seconds: int | None = None,
    min_pass_rate: float = 0.8,
    allow_failed_cases: bool = False,
    allow_non_succeeded_model: bool = False,
    attach_round_eval: bool = False,
    candidate_eval_report: Path | None = None,
    candidate_eval_recipe_path: Path | None = None,
    candidate_eval_mode: str = "run",
    candidate_eval_run_id: str | None = None,
) -> dict[str, Any]:
    if train_mode not in TRAIN_MODES:
        raise ValueError(f"train_mode must be one of: {', '.join(sorted(TRAIN_MODES))}")
    if candidate_eval_mode not in EVAL_MODES:
        raise ValueError(f"candidate_eval_mode must be one of: {', '.join(sorted(EVAL_MODES))}")
    if attach_round_eval and candidate_eval_report is not None:
        raise ValueError("attach_round_eval and candidate_eval_report cannot be used together")
    if candidate_eval_recipe_path is not None and (attach_round_eval or candidate_eval_report is not None):
        raise ValueError("candidate_eval_recipe cannot be combined with attach_round_eval or candidate_eval_report")
    if (attach_round_eval or candidate_eval_report is not None or candidate_eval_recipe_path is not None) and not model_id:
        raise ValueError("model_id is required when attaching an eval report")

    round_dir.mkdir(parents=True, exist_ok=True)
    recorder = _RoundRecorder(round_dir, workspace_id)
    data_run_dir = round_dir / "data_run"
    train_result: dict[str, Any] | None = None
    model: dict[str, Any] | None = None
    promotion: dict[str, Any] | None = None
    candidate_eval_result: dict[str, Any] | None = None
    candidate_eval_run_dir: Path | None = None
    candidate_eval_report_path: Path | None = None
    candidate_eval_report_data: dict[str, Any] | None = None
    candidate_case_report: dict[str, Any] | None = None

    try:
        tasks = _stage_ingest(recorder, workspace_id, seed_path, data_run_dir)
        traces = _stage_rollout(recorder, agent, tasks, data_run_dir)
        eval_report = _stage_eval(recorder, workspace_id, suite_path, traces, data_run_dir)
        case_report = _stage_analyze(recorder, data_run_dir)
        _stage_export(recorder, data_run_dir, tasks)

        train_run_dir: Path | None = None
        if recipe_path is not None:
            train_result, train_run_dir = _stage_train(
                recorder,
                workspace_id,
                recipe_path,
                train_mode,
                round_dir,
                train_run_id=train_run_id,
                cwd=cwd,
                env=env,
                timeout_seconds=timeout_seconds,
            )

        if candidate_eval_recipe_path is not None:
            candidate_eval_result, candidate_eval_run_dir, candidate_eval_report_path = _stage_candidate_eval(
                recorder,
                workspace_id,
                candidate_eval_recipe_path,
                candidate_eval_mode,
                round_dir,
                registry_root=registry_root or Path("."),
                eval_run_id=candidate_eval_run_id,
                cwd=cwd,
                env=env,
                timeout_seconds=timeout_seconds,
            )
            if candidate_eval_report_path is not None and candidate_eval_report_path.exists():
                candidate_eval_report_data = read_json(candidate_eval_report_path)
                candidate_case_report = _stage_candidate_analyze(recorder, candidate_eval_run_dir, candidate_eval_result)

        if model_id and train_result is not None and train_run_dir is not None:
            model, promotion = _stage_registry(
                recorder,
                root=registry_root or Path("."),
                model_id=model_id,
                train_run_dir=train_run_dir,
                model_path=model_path,
                allow_non_succeeded=allow_non_succeeded_model or train_mode == "plan",
                eval_report=data_run_dir / "eval_report.json" if attach_round_eval else candidate_eval_report,
                eval_run_dir=candidate_eval_run_dir if candidate_eval_report_path is not None else None,
                min_pass_rate=min_pass_rate,
                allow_failed_cases=allow_failed_cases,
            )

        final_eval_report = candidate_eval_report_data or eval_report
        final_case_report = candidate_case_report or case_report
        _stage_plan_next(recorder, final_eval_report, final_case_report, promotion)
        quality_status = _quality_status(final_eval_report, promotion)
        recorder.complete(
            status="succeeded",
            quality_status=quality_status,
            summary={
                "tasks": len(tasks),
                "traces": len(traces),
                "initial_pass_rate": eval_report.get("pass_rate"),
                "initial_failed": eval_report.get("failed"),
                "pass_rate": final_eval_report.get("pass_rate"),
                "failed": final_eval_report.get("failed"),
                "model_id": model.get("model_id") if model else None,
                "promotion_decision": promotion.get("decision") if promotion else None,
                "candidate_eval_run_id": candidate_eval_result.get("eval_run_id") if candidate_eval_result else None,
            },
        )
        return recorder.manifest
    except Exception as exc:
        recorder.fail(exc)
        raise


def _stage_ingest(recorder: "_RoundRecorder", workspace_id: str, seed_path: Path, data_run_dir: Path) -> list[dict[str, Any]]:
    stage = recorder.start_stage("ingest", {"input": str(seed_path)})
    records = read_jsonl(seed_path)
    accepted: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for record in records:
        validation_errors = validate_task_sample(record)
        if record.get("workspace_id") != workspace_id:
            validation_errors.append("workspace_id does not match workspace")
        if validation_errors:
            errors.append({"record": record, "errors": validation_errors})
        else:
            accepted.append(record)

    store = LocalRunStore(data_run_dir)
    store.init(workspace_id, str(seed_path))
    store.save_records("tasks", "tasks.jsonl", accepted)
    store.save_records("ingest_errors", "ingest_errors.jsonl", errors)
    store.save_report(
        "ingest_summary",
        "ingest_summary.json",
        {"input": str(seed_path), "accepted": len(accepted), "errors": len(errors)},
    )
    recorder.finish_stage(
        stage,
        "succeeded" if not errors else "failed",
        outputs={
            "data_run": _relative(data_run_dir, recorder.round_dir),
            "tasks": _relative(data_run_dir / "tasks.jsonl", recorder.round_dir),
            "ingest_errors": _relative(data_run_dir / "ingest_errors.jsonl", recorder.round_dir),
        },
        metrics={"accepted": len(accepted), "errors": len(errors)},
    )
    if errors:
        raise ValueError(f"seed input has invalid records: {len(errors)}")
    return accepted


def _stage_rollout(
    recorder: "_RoundRecorder",
    agent: str,
    tasks: list[dict[str, Any]],
    data_run_dir: Path,
) -> list[dict[str, Any]]:
    stage = recorder.start_stage("rollout", {"agent": agent})
    store = LocalRunStore(data_run_dir)
    traces = rollout_tasks(tasks, agent=agent)
    store.save_records("traces", "traces.jsonl", traces)
    store.save_report("rollout_summary", "rollout_summary.json", {"agent": agent, "traces": len(traces)})
    recorder.finish_stage(
        stage,
        "succeeded",
        outputs={"traces": _relative(data_run_dir / "traces.jsonl", recorder.round_dir)},
        metrics={"traces": len(traces)},
    )
    return traces


def _stage_eval(
    recorder: "_RoundRecorder",
    workspace_id: str,
    suite_path: Path,
    traces: list[dict[str, Any]],
    data_run_dir: Path,
) -> dict[str, Any]:
    stage = recorder.start_stage("eval", {"suite": str(suite_path)})
    store = LocalRunStore(data_run_dir)
    suite = read_jsonl(suite_path)
    results, report, accepted, rejected = evaluate_traces(traces, suite)
    store.save_records("eval_results", "eval_results.jsonl", results)
    store.save_report("eval_report", "eval_report.json", report)
    store.save_records("accepted", "accepted.jsonl", accepted)
    store.save_records("rejected", "rejected.jsonl", rejected)
    recorder.finish_stage(
        stage,
        "succeeded",
        outputs={
            "eval_report": _relative(data_run_dir / "eval_report.json", recorder.round_dir),
            "eval_results": _relative(data_run_dir / "eval_results.jsonl", recorder.round_dir),
            "accepted": _relative(data_run_dir / "accepted.jsonl", recorder.round_dir),
            "rejected": _relative(data_run_dir / "rejected.jsonl", recorder.round_dir),
        },
        metrics={
            "total": report.get("total"),
            "passed": report.get("passed"),
            "failed": report.get("failed"),
            "pass_rate": report.get("pass_rate"),
            "workspace_id": workspace_id,
        },
    )
    return report


def _stage_analyze(recorder: "_RoundRecorder", data_run_dir: Path) -> dict[str, Any]:
    stage = recorder.start_stage("analyze")
    store = LocalRunStore(data_run_dir)
    report = analyze_cases(store.load_records("eval_results.jsonl"), store.load_records("traces.jsonl"))
    store.save_report("case_report", "case_report.json", report)
    markdown_path = store.path("case_report.md")
    markdown_path.write_text(render_case_report_markdown(report), encoding="utf-8", newline="\n")
    store.update_artifact("case_report_md", "case_report.md")
    recorder.finish_stage(
        stage,
        "succeeded",
        outputs={
            "case_report": _relative(data_run_dir / "case_report.json", recorder.round_dir),
            "case_report_md": _relative(markdown_path, recorder.round_dir),
        },
        metrics={
            "failed_cases": report.get("total_failed_cases"),
            "recommended_actions": report.get("recommended_action_counts", {}),
        },
    )
    return report


def _stage_export(recorder: "_RoundRecorder", data_run_dir: Path, tasks: list[dict[str, Any]]) -> None:
    stage = recorder.start_stage("export")
    store = LocalRunStore(data_run_dir)
    accepted_path = data_run_dir / "accepted.jsonl"
    traces = store.load_records("accepted.jsonl" if accepted_path.exists() else "traces.jsonl")
    case_report = read_json(data_run_dir / "case_report.json")
    sft_path = store.path("sft.jsonl")
    preference_path = store.path("preference.jsonl")
    rl_path = store.path("rl_prompts.jsonl")
    write_jsonl(sft_path, export_sft(traces))
    write_jsonl(preference_path, export_preference(case_report.get("cases", [])))
    write_jsonl(rl_path, export_rl_prompts(tasks))
    store.update_artifact("sft", "sft.jsonl")
    store.update_artifact("preference", "preference.jsonl")
    store.update_artifact("rl_prompts", "rl_prompts.jsonl")
    dataset_versions = _write_round_dataset_versions(recorder, data_run_dir)
    recorder.manifest["datasets"] = {
        name: dataset_summary(manifest) for name, manifest in sorted(dataset_versions.items())
    }
    recorder.finish_stage(
        stage,
        "succeeded",
        outputs={
            "sft": _relative(sft_path, recorder.round_dir),
            "preference": _relative(preference_path, recorder.round_dir),
            "rl_prompts": _relative(rl_path, recorder.round_dir),
            "dataset_versions": _relative(data_run_dir / "dataset_versions.json", recorder.round_dir),
        },
        metrics={
            "sft_records": _count_jsonl(sft_path),
            "preference_records": _count_jsonl(preference_path),
            "rl_prompt_records": _count_jsonl(rl_path),
            "dataset_versions": len(dataset_versions),
            "dataset_schema_failed": sum(
                1 for item in dataset_versions.values() if item.get("schema_status") == "failed"
            ),
        },
    )


def _stage_train(
    recorder: "_RoundRecorder",
    workspace_id: str,
    recipe_path: Path,
    train_mode: str,
    round_dir: Path,
    *,
    train_run_id: str | None,
    cwd: Path | None,
    env: dict[str, str] | None,
    timeout_seconds: int | None,
) -> tuple[dict[str, Any], Path | None]:
    stage = recorder.start_stage("train", {"recipe": str(recipe_path), "mode": train_mode})
    manifest = build_train_dry_run(recipe_path, workspace=workspace_id)
    dry_run_path = round_dir / "train_dry_run.json"
    write_json(dry_run_path, manifest)

    outputs = {"train_dry_run": _relative(dry_run_path, recorder.round_dir)}
    if train_mode == "dry-run":
        recorder.finish_stage(
            stage,
            "succeeded" if manifest["preflight"]["status"] != "failed" else "failed",
            outputs=outputs,
            metrics={"preflight": manifest["preflight"]["status"]},
        )
        if manifest["preflight"]["status"] == "failed":
            raise ValueError("training preflight failed")
        return manifest, None

    if manifest["preflight"]["status"] == "failed":
        recorder.finish_stage(stage, "failed", outputs=outputs, metrics={"preflight": "failed"})
        raise ValueError("training preflight failed")

    run_id = train_run_id or create_train_run_id(manifest)
    train_run_dir = round_dir / "train_runs" / run_id
    if train_mode == "plan":
        result = plan_train_run(manifest, train_run_dir)
    else:
        result = run_train_manifest(manifest, train_run_dir, cwd=cwd, env=env, timeout_seconds=timeout_seconds)

    outputs["train_run"] = _relative(train_run_dir, recorder.round_dir)
    recorder.finish_stage(
        stage,
        result.get("status", "unknown"),
        outputs=outputs,
        metrics={
            "backend": result.get("backend"),
            "stage": result.get("stage"),
            "return_code": result.get("return_code"),
            "duration_seconds": result.get("duration_seconds"),
        },
    )
    if result.get("status") not in {"planned", "succeeded"}:
        raise ValueError(f"train run failed: {result.get('status')}")
    return result, train_run_dir


def _stage_candidate_eval(
    recorder: "_RoundRecorder",
    workspace_id: str,
    recipe_path: Path,
    eval_mode: str,
    round_dir: Path,
    *,
    registry_root: Path,
    eval_run_id: str | None,
    cwd: Path | None,
    env: dict[str, str] | None,
    timeout_seconds: int | None,
) -> tuple[dict[str, Any], Path | None, Path | None]:
    stage = recorder.start_stage("candidate_eval", {"recipe": str(recipe_path), "mode": eval_mode})
    manifest = build_eval_dry_run(recipe_path, workspace=workspace_id, registry_root=registry_root)
    dry_run_path = round_dir / "candidate_eval_dry_run.json"
    write_json(dry_run_path, manifest)
    outputs = {"candidate_eval_dry_run": _relative(dry_run_path, recorder.round_dir)}

    if eval_mode == "dry-run":
        recorder.finish_stage(
            stage,
            "succeeded" if manifest["preflight"]["status"] != "failed" else "failed",
            outputs=outputs,
            metrics={
                "preflight": manifest["preflight"]["status"],
                "datasets": len((manifest.get("dataset_resolution") or {}).get("datasets", [])),
            },
        )
        if manifest["preflight"]["status"] == "failed":
            raise ValueError("candidate eval preflight failed")
        return manifest, None, None

    if manifest["preflight"]["status"] == "failed":
        recorder.finish_stage(stage, "failed", outputs=outputs, metrics={"preflight": "failed"})
        raise ValueError("candidate eval preflight failed")

    run_id = eval_run_id or create_eval_run_id(manifest)
    eval_run_dir = round_dir / "eval_runs" / run_id
    if eval_mode == "plan":
        result = plan_eval_run(manifest, eval_run_dir)
    else:
        result = run_eval_manifest(manifest, eval_run_dir, cwd=cwd, env=env, timeout_seconds=timeout_seconds)

    report_path = eval_run_dir / "eval_report.json"
    outputs["candidate_eval_run"] = _relative(eval_run_dir, recorder.round_dir)
    if report_path.exists():
        outputs["candidate_eval_report"] = _relative(report_path, recorder.round_dir)
    recorder.finish_stage(
        stage,
        result.get("status", "unknown"),
        outputs=outputs,
        metrics={
            "backend": result.get("backend"),
            "stage": result.get("stage"),
            "quality_status": result.get("quality_status"),
            "return_code": result.get("return_code"),
            "duration_seconds": result.get("duration_seconds"),
            "datasets": len((manifest.get("dataset_resolution") or {}).get("datasets", [])),
        },
    )
    if result.get("status") not in {"planned", "succeeded"}:
        raise ValueError(f"candidate eval failed: {result.get('status')}")
    return result, eval_run_dir, report_path if report_path.exists() else None


def _stage_candidate_analyze(
    recorder: "_RoundRecorder",
    eval_run_dir: Path | None,
    eval_result: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if eval_run_dir is None or eval_result is None:
        return None
    eval_results_path = eval_run_dir / "eval_results.jsonl"
    traces_path = _candidate_eval_trace_path(eval_result)
    if not eval_results_path.exists() or traces_path is None or not traces_path.exists():
        return None

    stage = recorder.start_stage("candidate_analyze", {"eval_run": _relative(eval_run_dir, recorder.round_dir)})
    report = analyze_cases(read_jsonl(eval_results_path), read_jsonl(traces_path))
    report_path = eval_run_dir / "case_report.json"
    markdown_path = eval_run_dir / "case_report.md"
    write_json(report_path, report)
    markdown_path.write_text(render_case_report_markdown(report), encoding="utf-8", newline="\n")
    recorder.finish_stage(
        stage,
        "succeeded",
        outputs={
            "candidate_case_report": _relative(report_path, recorder.round_dir),
            "candidate_case_report_md": _relative(markdown_path, recorder.round_dir),
        },
        metrics={
            "failed_cases": report.get("total_failed_cases"),
            "recommended_actions": report.get("recommended_action_counts", {}),
        },
    )
    return report


def _stage_registry(
    recorder: "_RoundRecorder",
    *,
    root: Path,
    model_id: str,
    train_run_dir: Path,
    model_path: str | None,
    allow_non_succeeded: bool,
    eval_report: Path | None,
    eval_run_dir: Path | None,
    min_pass_rate: float,
    allow_failed_cases: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    stage = recorder.start_stage("registry", {"model_id": model_id, "root": str(root)})
    registered_datasets = 0
    dataset_version_ids: list[str] = []
    for dataset in (recorder.manifest.get("datasets") or {}).values():
        if isinstance(dataset, dict):
            full_manifest = _load_dataset_manifest(data_run_dir=recorder.round_dir / "data_run", summary=dataset)
            if full_manifest is not None:
                registered = register_dataset_manifest(root, full_manifest)
                registered_datasets += 1
                dataset_version_ids.append(str(registered["dataset_version_id"]))
    index_train_run(root, train_run_dir)
    model = register_model(
        root,
        model_id=model_id,
        run_dir=train_run_dir,
        model_path=model_path,
        allow_non_succeeded=allow_non_succeeded,
        dataset_version_ids=dataset_version_ids,
    )
    promotion = None
    if eval_run_dir is not None:
        if (eval_run_dir / "eval_report.json").exists():
            attached = attach_eval_run(
                root,
                model_id=model_id,
                eval_run_dir=eval_run_dir,
                min_pass_rate=min_pass_rate,
                allow_failed_cases=allow_failed_cases,
            )
            model = attached["model"]
            promotion = attached["promotion"]
        else:
            index_eval_run(root, eval_run_dir)
    elif eval_report is not None:
        attached = attach_eval_report(
            root,
            model_id=model_id,
            eval_report_path=eval_report,
            min_pass_rate=min_pass_rate,
            allow_failed_cases=allow_failed_cases,
        )
        model = attached["model"]
        promotion = attached["promotion"]
    recorder.finish_stage(
        stage,
        "succeeded",
        outputs={"registry": str(registry_path(root))},
        metrics={
            "model_id": model.get("model_id"),
            "model_status": model.get("status"),
            "promotion_decision": promotion.get("decision") if promotion else None,
            "registered_datasets": registered_datasets,
        },
    )
    return model, promotion


def _stage_plan_next(
    recorder: "_RoundRecorder",
    eval_report: dict[str, Any],
    case_report: dict[str, Any],
    promotion: dict[str, Any] | None,
) -> dict[str, Any]:
    stage = recorder.start_stage("plan_next")
    plan = build_iteration_plan(
        round_manifest=recorder.manifest,
        eval_report=eval_report,
        case_report=case_report,
        promotion=promotion,
    )
    outputs = write_iteration_plan(recorder.round_dir, plan)
    recorder.finish_stage(
        stage,
        "succeeded",
        outputs=outputs,
        metrics={
            "quality_status": plan.get("quality_status"),
            "action_count": (plan.get("summary") or {}).get("action_count"),
        },
    )
    return plan


class _RoundRecorder:
    def __init__(self, round_dir: Path, workspace_id: str):
        now = _now()
        self.round_dir = round_dir
        self.manifest: dict[str, Any] = {
            "round_id": round_dir.name,
            "workspace_id": workspace_id,
            "status": "running",
            "quality_status": "unknown",
            "created_at": now,
            "updated_at": now,
            "started_at": now,
            "stages": [],
            "artifacts": {},
            "summary": {},
        }
        self._write_manifest()
        (self.round_dir / ROUND_EVENTS_FILENAME).write_text("", encoding="utf-8", newline="\n")
        self.event("round_started", {"workspace_id": workspace_id})

    def start_stage(self, name: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        stage = {"name": name, "status": "running", "started_at": _now(), "inputs": payload or {}}
        self.event("stage_started", {"stage": name, **(payload or {})})
        return stage

    def finish_stage(
        self,
        stage: dict[str, Any],
        status: str,
        *,
        outputs: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        stage["status"] = status
        stage["finished_at"] = _now()
        if outputs:
            stage["outputs"] = outputs
            self.manifest.setdefault("artifacts", {}).update(outputs)
        if metrics:
            stage["metrics"] = metrics
        self.manifest.setdefault("stages", []).append(stage)
        self._touch()
        self.event("stage_finished", {"stage": stage["name"], "status": status})

    def complete(self, *, status: str, quality_status: str, summary: dict[str, Any]) -> None:
        self.manifest["status"] = status
        self.manifest["quality_status"] = quality_status
        self.manifest["summary"] = summary
        self.manifest["completed_at"] = _now()
        self._touch()
        self.event("round_finished", {"status": status, "quality_status": quality_status})

    def fail(self, exc: Exception) -> None:
        self.manifest["status"] = "failed"
        self.manifest["quality_status"] = "unknown"
        self.manifest["error"] = {"type": exc.__class__.__name__, "message": str(exc)}
        self.manifest["completed_at"] = _now()
        self._touch()
        self.event("round_failed", self.manifest["error"])

    def event(self, event_type: str, payload: dict[str, Any]) -> None:
        event = {"time": _now(), "event": event_type, **payload}
        path = self.round_dir / ROUND_EVENTS_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(event, ensure_ascii=False, sort_keys=True))
            f.write("\n")

    def _touch(self) -> None:
        self.manifest["updated_at"] = _now()
        self._write_manifest()

    def _write_manifest(self) -> None:
        write_json(self.round_dir / ROUND_FILENAME, self.manifest)


def _quality_status(eval_report: dict[str, Any], promotion: dict[str, Any] | None) -> str:
    if promotion is not None:
        return str(promotion.get("decision") or "unknown")
    return "passed" if int(eval_report.get("failed", 0)) == 0 else "needs_iteration"


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _count_jsonl(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _write_round_dataset_versions(
    recorder: "_RoundRecorder",
    data_run_dir: Path,
) -> dict[str, dict[str, Any]]:
    specs = {
        "tasks": ("task", "tasks.jsonl"),
        "traces": ("trace", "traces.jsonl"),
        "eval_results": ("eval_result", "eval_results.jsonl"),
        "eval_report": ("eval_report", "eval_report.json"),
        "accepted": ("trace", "accepted.jsonl"),
        "rejected": ("trace", "rejected.jsonl"),
        "sft": ("sft", "sft.jsonl"),
        "preference": ("preference", "preference.jsonl"),
        "rl_prompts": ("rl_prompt", "rl_prompts.jsonl"),
    }
    manifests: dict[str, dict[str, Any]] = {}
    for name, (kind, filename) in specs.items():
        path = data_run_dir / filename
        if not path.exists():
            continue
        manifest = build_dataset_version(
            path,
            kind=kind,
            dataset_id=f"{recorder.round_dir.name}-{name}",
            workspace_id=str(recorder.manifest.get("workspace_id") or ""),
            source={
                "round_id": recorder.manifest.get("round_id"),
                "stage": "export",
                "artifact": name,
            },
        )
        manifests[name] = manifest
    write_dataset_collection(data_run_dir / "dataset_versions.json", list(manifests.values()))
    return manifests


def _load_dataset_manifest(data_run_dir: Path, summary: dict[str, Any]) -> dict[str, Any] | None:
    collection_path = data_run_dir / "dataset_versions.json"
    if not collection_path.exists():
        return None
    collection = read_json(collection_path)
    dataset_version_id = summary.get("dataset_version_id")
    for manifest in collection.get("datasets", []):
        if isinstance(manifest, dict) and manifest.get("dataset_version_id") == dataset_version_id:
            return manifest
    return None


def _candidate_eval_trace_path(eval_result: dict[str, Any]) -> Path | None:
    execution = eval_result.get("execution") if isinstance(eval_result.get("execution"), dict) else {}
    if not execution:
        source_manifest = eval_result.get("source_manifest") if isinstance(eval_result.get("source_manifest"), dict) else {}
        execution = source_manifest.get("execution") if isinstance(source_manifest.get("execution"), dict) else {}
    traces = execution.get("traces")
    if not traces:
        return None
    return Path(str(traces))


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()).strip("-").lower()
    return slug or "round"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
