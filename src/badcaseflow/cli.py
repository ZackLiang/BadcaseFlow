from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .actions import ACTION_STATUSES, list_actions as list_iteration_actions, update_action
from .analyzer import analyze_cases, render_case_report_markdown
from .datasets import SUPPORTED_DATASET_KINDS, build_dataset_version, write_dataset_manifest
from .deployment import SERVING_BACKENDS, build_deployment_plan, render_deployment_script
from .evaluation import evaluate_traces
from .evals import list_eval_adapters
from .evals.harness import create_eval_run_id, plan_eval_run, run_eval_manifest
from .exporters import export_preference, export_rl_prompts, export_sft
from .gsm8k import DEFAULT_SOURCE_NAME, DEFAULT_SUITE_ID, DEFAULT_WORKSPACE_ID, prepare_gsm8k
from .io_utils import read_json, read_jsonl, write_json, write_jsonl
from .iteration import build_iteration_plan_from_round, write_iteration_plan
from .jobs import collect_job, execute_job, get_job_status, list_jobs, read_job_log, submit_job
from .launcher import build_launch_plan, render_launch_script
from .registry import (
    attach_eval_run,
    attach_eval_report,
    decide_promotion,
    get_dataset,
    get_deployment,
    get_model,
    get_model_lineage,
    index_eval_run,
    index_train_run,
    list_artifacts,
    list_datasets,
    list_deployments,
    list_evaluations,
    list_models,
    list_promotions,
    register_dataset,
    register_deployment,
    register_model,
    registry_path,
)
from .remote import add_remote, build_remote_plan, list_remotes, render_plan_script
from .recipes import build_eval_dry_run, build_train_dry_run, save_dry_run_manifest
from .rollout import rollout_tasks
from .rounds import list_rounds, load_round, run_local_round
from .schemas import validate_agent_trace, validate_task_sample
from .store import LocalRunStore
from .training import list_training_adapters
from .training.harness import create_train_run_id, parse_env_overrides, plan_train_run, run_train_manifest
from .workbench import DEFAULT_WORKBENCH_OUTPUT, build_workbench
from .workspace import init_workspace, load_workspace, remotes_path


# CLI 只负责参数解析、路径编排和结果展示；核心业务逻辑放在独立模块中，
# 方便以后接入 Web 工作台或其他执行入口。
def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    # CLI 边界统一转成可读错误，避免把底层 traceback 直接展示给使用者。
    except Exception as exc:  # pragma: no cover - CLI boundary
        print(f"error: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bcf", description="BadcaseFlow local CLI")
    sub = parser.add_subparsers(required=True)

    init = sub.add_parser("init-workspace", help="create local BadcaseFlow workspace config")
    init.add_argument("--workspace", required=True)
    init.add_argument("--root", default=Path("."), type=Path)
    init.add_argument("--force", action="store_true")
    init.set_defaults(func=cmd_init_workspace)

    version = sub.add_parser("version", help="show BadcaseFlow version")
    version.add_argument("--json", action="store_true")
    version.set_defaults(func=cmd_version)

    doctor = sub.add_parser("doctor", help="check local development environment")
    doctor.add_argument("--root", default=Path("."), type=Path)
    doctor.add_argument("--strict", action="store_true")
    doctor.add_argument("--json", action="store_true")
    doctor.set_defaults(func=cmd_doctor)

    ingest = sub.add_parser("ingest", help="import task samples into a run")
    ingest.add_argument("--workspace", required=True)
    ingest.add_argument("--input", required=True, type=Path)
    ingest.add_argument("--run", required=True, type=Path)
    ingest.set_defaults(func=cmd_ingest)

    import_traces = sub.add_parser("import-traces", help="import external agent traces into a run")
    import_traces.add_argument("--workspace", required=True)
    import_traces.add_argument("--input", required=True, type=Path)
    import_traces.add_argument("--run", required=True, type=Path)
    import_traces.set_defaults(func=cmd_import_traces)

    inspect = sub.add_parser("inspect", help="show run summary")
    inspect.add_argument("--run", required=True, type=Path)
    inspect.set_defaults(func=cmd_inspect)

    rollout = sub.add_parser("rollout", help="run mock agent rollout")
    rollout.add_argument("--workspace", required=True)
    rollout.add_argument("--run", required=True, type=Path)
    rollout.add_argument("--agent", default="mock", choices=["mock", "mock-buggy"])
    rollout.set_defaults(func=cmd_rollout)

    evaluate = sub.add_parser("eval", help="evaluate traces")
    evaluate.add_argument("--workspace", required=True)
    evaluate.add_argument("--run", required=True, type=Path)
    evaluate.add_argument("--suite", required=True, type=Path)
    evaluate.set_defaults(func=cmd_eval)

    evals = sub.add_parser("evals", help="evaluation adapter commands")
    evals_sub = evals.add_subparsers(required=True)
    eval_adapters = evals_sub.add_parser("adapters", aliases=["list-adapters"], help="list eval adapters")
    eval_adapters.set_defaults(func=cmd_evals_adapters)

    eval_dry = evals_sub.add_parser("dry-run", help="render and preflight an eval recipe")
    eval_dry.add_argument("--workspace", required=False)
    eval_dry.add_argument("--recipe", required=True, type=Path)
    eval_dry.add_argument("--output-root", default=Path("runs"), type=Path)
    eval_dry.add_argument("--registry-root", required=False, type=Path)
    eval_dry.set_defaults(func=cmd_evals_dry_run)

    eval_run = evals_sub.add_parser("run", help="execute an eval recipe through the local harness")
    eval_run.add_argument("--workspace", required=False)
    eval_run.add_argument("--recipe", required=True, type=Path)
    eval_run.add_argument("--output-root", default=Path("runs"), type=Path)
    eval_run.add_argument("--run-id", required=False)
    eval_run.add_argument("--cwd", default=Path("."), type=Path)
    eval_run.add_argument("--timeout-seconds", required=False, type=int)
    eval_run.add_argument("--env", action="append", default=[])
    eval_run.add_argument("--registry-root", required=False, type=Path)
    eval_run.add_argument("--allow-failed-preflight", action="store_true")
    eval_run.add_argument("--plan-only", action="store_true")
    eval_run.add_argument("--fail-on-blocked", action="store_true")
    eval_run.set_defaults(func=cmd_evals_run)

    eval_inspect = evals_sub.add_parser("inspect", help="show eval run status and metrics")
    eval_inspect.add_argument("--run", required=True, type=Path)
    eval_inspect.add_argument("--tail", default=5, type=int)
    eval_inspect.set_defaults(func=cmd_evals_inspect)

    eval_list = evals_sub.add_parser("list", help="list local eval runs")
    eval_list.add_argument("--root", default=Path("runs/eval_runs"), type=Path)
    eval_list.set_defaults(func=cmd_evals_list)

    analyze = sub.add_parser("analyze-cases", help="analyze failed eval cases")
    analyze.add_argument("--workspace", required=True)
    analyze.add_argument("--run", required=True, type=Path)
    analyze.set_defaults(func=cmd_analyze_cases)

    export = sub.add_parser("export", help="export train/eval datasets")
    export_sub = export.add_subparsers(required=True)
    sft = export_sub.add_parser("sft", help="export accepted traces as SFT data")
    sft.add_argument("--workspace", required=True)
    sft.add_argument("--run", required=True, type=Path)
    sft.add_argument("--output", required=True, type=Path)
    sft.set_defaults(func=cmd_export_sft)
    pref = export_sub.add_parser("preference", help="export failed cases as preference candidates")
    pref.add_argument("--workspace", required=True)
    pref.add_argument("--run", required=True, type=Path)
    pref.add_argument("--output", required=True, type=Path)
    pref.set_defaults(func=cmd_export_preference)
    rl = export_sub.add_parser("rl-prompt", help="export tasks as RL prompt data")
    rl.add_argument("--workspace", required=True)
    rl.add_argument("--run", required=True, type=Path)
    rl.add_argument("--output", required=True, type=Path)
    rl.set_defaults(func=cmd_export_rl_prompt)

    train = sub.add_parser("train", help="training adapter commands")
    train_sub = train.add_subparsers(required=True)
    dry = train_sub.add_parser("dry-run", help="render and preflight a training recipe")
    dry.add_argument("--workspace", required=False)
    dry.add_argument("--recipe", required=True, type=Path)
    dry.add_argument("--output-root", default=Path("runs"), type=Path)
    dry.add_argument("--registry-root", required=False, type=Path)
    dry.set_defaults(func=cmd_train_dry_run)
    run = train_sub.add_parser("run", help="execute a training recipe through the local harness")
    run.add_argument("--workspace", required=False)
    run.add_argument("--recipe", required=True, type=Path)
    run.add_argument("--output-root", default=Path("runs"), type=Path)
    run.add_argument("--run-id", required=False)
    run.add_argument("--cwd", default=Path("."), type=Path)
    run.add_argument("--timeout-seconds", required=False, type=int)
    run.add_argument("--env", action="append", default=[])
    run.add_argument("--registry-root", required=False, type=Path)
    run.add_argument("--allow-failed-preflight", action="store_true")
    run.add_argument("--plan-only", action="store_true")
    run.set_defaults(func=cmd_train_run)
    train_inspect = train_sub.add_parser("inspect", help="show training run status and metric summary")
    train_inspect.add_argument("--run", required=True, type=Path)
    train_inspect.add_argument("--tail", default=5, type=int)
    train_inspect.set_defaults(func=cmd_train_inspect)
    train_list = train_sub.add_parser("list", help="list local training runs")
    train_list.add_argument("--root", default=Path("runs/train_runs"), type=Path)
    train_list.set_defaults(func=cmd_train_list)
    adapters = train_sub.add_parser(
        "adapters",
        aliases=["list-adapters"],
        help="list supported training adapters and stages",
    )
    adapters.set_defaults(func=cmd_train_adapters)

    compare = sub.add_parser("compare", help="compare two eval reports")
    compare.add_argument("--before", required=True, type=Path)
    compare.add_argument("--after", required=True, type=Path)
    compare.set_defaults(func=cmd_compare)

    promote = sub.add_parser("promote", help="promotion gate commands")
    promote_sub = promote.add_subparsers(required=True)
    promote_dry = promote_sub.add_parser("dry-run", help="check if a model can be promoted")
    promote_dry.add_argument("--model", required=True)
    promote_dry.add_argument("--eval-report", required=True, type=Path)
    promote_dry.add_argument("--min-pass-rate", default=0.8, type=float)
    promote_dry.add_argument("--allow-failed-cases", action="store_true")
    promote_dry.set_defaults(func=cmd_promote_dry_run)

    data = sub.add_parser("data", help="data preparation commands")
    data_sub = data.add_subparsers(required=True)
    data_gsm8k = data_sub.add_parser("prepare-gsm8k", help="prepare GSM8K-style math data")
    data_gsm8k.add_argument("--input", required=True, type=Path)
    data_gsm8k.add_argument("--val-input", required=False, type=Path)
    data_gsm8k.add_argument("--output-dir", required=True, type=Path)
    data_gsm8k.add_argument("--workspace", default=DEFAULT_WORKSPACE_ID)
    data_gsm8k.add_argument("--source-name", default=DEFAULT_SOURCE_NAME)
    data_gsm8k.add_argument("--suite-id", default=DEFAULT_SUITE_ID)
    data_gsm8k.add_argument("--train-ratio", default=0.8, type=float)
    data_gsm8k.add_argument("--limit", required=False, type=int)
    data_gsm8k.add_argument("--val-limit", required=False, type=int)
    data_gsm8k.add_argument("--parquet", action="store_true")
    data_gsm8k.add_argument("--register", action="store_true")
    data_gsm8k.add_argument("--registry-root", default=Path("."), type=Path)
    data_gsm8k.set_defaults(func=cmd_data_prepare_gsm8k)

    dataset = sub.add_parser("dataset", help="dataset version commands")
    dataset_sub = dataset.add_subparsers(required=True)
    dataset_inspect = dataset_sub.add_parser("inspect", help="inspect a dataset file and validate its schema")
    dataset_inspect.add_argument("--path", required=True, type=Path)
    dataset_inspect.add_argument("--kind", required=True, choices=SUPPORTED_DATASET_KINDS)
    dataset_inspect.add_argument("--dataset-id", required=False)
    dataset_inspect.add_argument("--workspace", required=False)
    dataset_inspect.add_argument("--description", required=False)
    dataset_inspect.add_argument("--output", required=False, type=Path)
    dataset_inspect.add_argument("--json", action="store_true")
    dataset_inspect.set_defaults(func=cmd_dataset_inspect)

    dataset_register = dataset_sub.add_parser("register", help="register a dataset version in the local registry")
    dataset_register.add_argument("--root", default=Path("."), type=Path)
    dataset_register.add_argument("--path", required=True, type=Path)
    dataset_register.add_argument("--kind", required=True, choices=SUPPORTED_DATASET_KINDS)
    dataset_register.add_argument("--dataset-id", required=False)
    dataset_register.add_argument("--workspace", required=False)
    dataset_register.add_argument("--description", required=False)
    dataset_register.set_defaults(func=cmd_dataset_register)

    dataset_list = dataset_sub.add_parser("list", help="list registered dataset versions")
    dataset_list.add_argument("--root", default=Path("."), type=Path)
    dataset_list.set_defaults(func=cmd_dataset_list)

    dataset_show = dataset_sub.add_parser("show", help="show a registered dataset version")
    dataset_show.add_argument("--root", default=Path("."), type=Path)
    dataset_show.add_argument("--dataset-version-id", required=True)
    dataset_show.set_defaults(func=cmd_dataset_show)

    registry = sub.add_parser("registry", help="local artifact and model registry commands")
    registry_sub = registry.add_subparsers(required=True)
    registry_index = registry_sub.add_parser("index-train-run", help="index artifacts from a train run")
    registry_index.add_argument("--root", default=Path("."), type=Path)
    registry_index.add_argument("--run", required=True, type=Path)
    registry_index.set_defaults(func=cmd_registry_index_train_run)

    registry_index_eval = registry_sub.add_parser("index-eval-run", help="index artifacts from an eval run")
    registry_index_eval.add_argument("--root", default=Path("."), type=Path)
    registry_index_eval.add_argument("--run", required=True, type=Path)
    registry_index_eval.set_defaults(func=cmd_registry_index_eval_run)

    registry_artifacts = registry_sub.add_parser("artifacts", help="list indexed artifacts")
    registry_artifacts.add_argument("--root", default=Path("."), type=Path)
    registry_artifacts.set_defaults(func=cmd_registry_artifacts)

    registry_register_model = registry_sub.add_parser("register-model", help="register a model candidate")
    registry_register_model.add_argument("--root", default=Path("."), type=Path)
    registry_register_model.add_argument("--run", required=True, type=Path)
    registry_register_model.add_argument("--model-id", required=True)
    registry_register_model.add_argument("--model-path", required=False)
    registry_register_model.add_argument("--description", required=False)
    registry_register_model.add_argument("--allow-non-succeeded", action="store_true")
    registry_register_model.set_defaults(func=cmd_registry_register_model)

    registry_models = registry_sub.add_parser("models", help="list registered models")
    registry_models.add_argument("--root", default=Path("."), type=Path)
    registry_models.set_defaults(func=cmd_registry_models)

    registry_show_model = registry_sub.add_parser("show-model", help="show a registered model")
    registry_show_model.add_argument("--root", default=Path("."), type=Path)
    registry_show_model.add_argument("--model-id", required=True)
    registry_show_model.set_defaults(func=cmd_registry_show_model)

    registry_attach_eval = registry_sub.add_parser("attach-eval", help="attach an eval report to a model candidate")
    registry_attach_eval.add_argument("--root", default=Path("."), type=Path)
    registry_attach_eval.add_argument("--model-id", required=True)
    registry_attach_eval.add_argument("--eval-report", required=True, type=Path)
    registry_attach_eval.add_argument("--min-pass-rate", default=0.8, type=float)
    registry_attach_eval.add_argument("--allow-failed-cases", action="store_true")
    registry_attach_eval.add_argument("--fail-on-blocked", action="store_true")
    registry_attach_eval.set_defaults(func=cmd_registry_attach_eval)

    registry_attach_eval_run = registry_sub.add_parser("attach-eval-run", help="attach an eval run to a model candidate")
    registry_attach_eval_run.add_argument("--root", default=Path("."), type=Path)
    registry_attach_eval_run.add_argument("--model-id", required=True)
    registry_attach_eval_run.add_argument("--eval-run", required=True, type=Path)
    registry_attach_eval_run.add_argument("--min-pass-rate", default=0.8, type=float)
    registry_attach_eval_run.add_argument("--allow-failed-cases", action="store_true")
    registry_attach_eval_run.add_argument("--fail-on-blocked", action="store_true")
    registry_attach_eval_run.set_defaults(func=cmd_registry_attach_eval_run)

    registry_evaluations = registry_sub.add_parser("evaluations", help="list model evaluations")
    registry_evaluations.add_argument("--root", default=Path("."), type=Path)
    registry_evaluations.set_defaults(func=cmd_registry_evaluations)

    registry_promotions = registry_sub.add_parser("promotions", help="list promotion decisions")
    registry_promotions.add_argument("--root", default=Path("."), type=Path)
    registry_promotions.set_defaults(func=cmd_registry_promotions)

    registry_lineage = registry_sub.add_parser("model-lineage", help="show model train/eval/promotion lineage")
    registry_lineage.add_argument("--root", default=Path("."), type=Path)
    registry_lineage.add_argument("--model-id", required=True)
    registry_lineage.set_defaults(func=cmd_registry_model_lineage)

    deploy = sub.add_parser("deploy", help="deployment planning commands")
    deploy_sub = deploy.add_subparsers(required=True)
    deploy_plan = deploy_sub.add_parser("plan", help="write a serving and canary deployment plan")
    deploy_plan.add_argument("--root", default=Path("."), type=Path)
    deploy_plan.add_argument("--model-id", required=True)
    deploy_plan.add_argument("--backend", default="vllm", choices=SERVING_BACKENDS)
    deploy_plan.add_argument("--mode", default="plan", choices=["plan", "runbook"])
    deploy_plan.add_argument("--deployment-id", required=False)
    deploy_plan.add_argument("--served-model-name", required=False)
    deploy_plan.add_argument("--host", default="0.0.0.0")
    deploy_plan.add_argument("--port", default=8000, type=int)
    deploy_plan.add_argument("--python", default="python")
    deploy_plan.add_argument("--endpoint", required=False)
    deploy_plan.add_argument("--canary-percent", default=5.0, type=float)
    deploy_plan.add_argument("--min-pass-rate", required=False, type=float)
    deploy_plan.add_argument("--previous-model-id", required=False)
    deploy_plan.add_argument("--rollback-endpoint", required=False)
    deploy_plan.add_argument("--allow-blocked", action="store_true")
    deploy_plan.add_argument("--output", required=False, type=Path)
    deploy_plan.add_argument("--register", action="store_true")
    deploy_plan.set_defaults(func=cmd_deploy_plan)

    deploy_list = deploy_sub.add_parser("list", help="list registered deployment plans")
    deploy_list.add_argument("--root", default=Path("."), type=Path)
    deploy_list.set_defaults(func=cmd_deploy_list)

    deploy_show = deploy_sub.add_parser("show", help="show a registered deployment plan")
    deploy_show.add_argument("--root", default=Path("."), type=Path)
    deploy_show.add_argument("--deployment-id", required=True)
    deploy_show.set_defaults(func=cmd_deploy_show)

    round_parser = sub.add_parser("round", help="flywheel round commands")
    round_sub = round_parser.add_subparsers(required=True)
    round_run = round_sub.add_parser("run-local", help="run a local flywheel round")
    round_run.add_argument("--workspace", required=True)
    round_run.add_argument("--round", required=True, type=Path)
    round_run.add_argument("--seed", default=Path("examples/datasets/seed_tasks.jsonl"), type=Path)
    round_run.add_argument("--suite", default=Path("examples/datasets/eval_tasks.jsonl"), type=Path)
    round_run.add_argument("--agent", default="mock", choices=["mock", "mock-buggy"])
    round_run.add_argument("--recipe", default=Path("examples/recipes/command_smoke.example.yaml"), type=Path)
    round_run.add_argument("--train-mode", default="dry-run", choices=["dry-run", "plan", "run"])
    round_run.add_argument("--run-id", required=False)
    round_run.add_argument("--cwd", default=Path("."), type=Path)
    round_run.add_argument("--timeout-seconds", required=False, type=int)
    round_run.add_argument("--env", action="append", default=[])
    round_run.add_argument("--registry-root", default=Path("."), type=Path)
    round_run.add_argument("--model-id", required=False)
    round_run.add_argument("--model-path", required=False)
    round_run.add_argument("--attach-round-eval", action="store_true")
    round_run.add_argument("--candidate-eval-report", required=False, type=Path)
    round_run.add_argument("--candidate-eval-recipe", required=False, type=Path)
    round_run.add_argument("--candidate-eval-mode", default="run", choices=["dry-run", "plan", "run"])
    round_run.add_argument("--candidate-eval-run-id", required=False)
    round_run.add_argument("--min-pass-rate", default=0.8, type=float)
    round_run.add_argument("--allow-failed-cases", action="store_true")
    round_run.add_argument("--allow-non-succeeded-model", action="store_true")
    round_run.add_argument("--fail-on-blocked", action="store_true")
    round_run.set_defaults(func=cmd_round_run_local)

    round_list = round_sub.add_parser("list", help="list flywheel rounds")
    round_list.add_argument("--root", default=Path("runs/rounds"), type=Path)
    round_list.set_defaults(func=cmd_round_list)

    round_inspect = round_sub.add_parser("inspect", help="show flywheel round summary")
    round_inspect.add_argument("--round", required=True, type=Path)
    round_inspect.add_argument("--json", action="store_true")
    round_inspect.set_defaults(func=cmd_round_inspect)

    round_plan = round_sub.add_parser("plan-next", help="write next-iteration action plan for a round")
    round_plan.add_argument("--round", required=True, type=Path)
    round_plan.add_argument("--output", required=False, type=Path)
    round_plan.set_defaults(func=cmd_round_plan_next)

    action = sub.add_parser("action", help="iteration action commands")
    action_sub = action.add_subparsers(required=True)
    action_list = action_sub.add_parser("list", help="list actions in an iteration plan")
    action_list.add_argument("--plan", required=True, type=Path)
    action_list.add_argument("--status", required=False, choices=sorted(ACTION_STATUSES))
    action_list.set_defaults(func=cmd_action_list)

    action_update = action_sub.add_parser("update", help="update action status, owner, or note")
    action_update.add_argument("--plan", required=True, type=Path)
    action_update.add_argument("--action-id", required=True)
    action_update.add_argument("--status", required=False, choices=sorted(ACTION_STATUSES))
    action_update.add_argument("--owner", required=False)
    action_update.add_argument("--note", required=False)
    action_update.set_defaults(func=cmd_action_update)

    launcher = sub.add_parser("launcher", help="local and ssh launch planning commands")
    launcher_sub = launcher.add_subparsers(required=True)
    launcher_plan = launcher_sub.add_parser("plan", help="write a local or ssh launch plan")
    launcher_plan.add_argument("--kind", required=True, choices=["train", "eval"])
    launcher_plan.add_argument("--recipe", required=True, type=Path)
    launcher_plan.add_argument("--launcher", default="local", choices=["local", "ssh"])
    launcher_plan.add_argument("--mode", default="dry-run", choices=["dry-run", "plan", "run"])
    launcher_plan.add_argument("--workspace", required=False)
    launcher_plan.add_argument("--output-root", default=Path("runs"), type=Path)
    launcher_plan.add_argument("--run-id", required=False)
    launcher_plan.add_argument("--python", default="python")
    launcher_plan.add_argument("--cwd", default=Path("."), type=Path)
    launcher_plan.add_argument("--config", default=Path(".badcaseflow/remotes.json"), type=Path)
    launcher_plan.add_argument("--target", required=False)
    launcher_plan.add_argument("--include", action="append", default=[], type=Path)
    launcher_plan.add_argument("--output", required=False, type=Path)
    launcher_plan.set_defaults(func=cmd_launcher_plan)

    job = sub.add_parser("job", help="launch job tracking commands")
    job_sub = job.add_subparsers(required=True)
    job_submit = job_sub.add_parser("submit", help="create a tracked job from a launch plan")
    job_submit.add_argument("--plan", required=True, type=Path)
    job_submit.add_argument("--root", default=Path("runs/jobs"), type=Path)
    job_submit.add_argument("--job-id", required=False)
    job_submit.add_argument("--execute", action="store_true")
    job_submit.add_argument("--timeout-seconds", required=False, type=int)
    job_submit.set_defaults(func=cmd_job_submit)

    job_run = job_sub.add_parser("run", help="execute a planned tracked job")
    job_run.add_argument("--job", required=True, type=Path)
    job_run.add_argument("--timeout-seconds", required=False, type=int)
    job_run.set_defaults(func=cmd_job_run)

    job_status = job_sub.add_parser("status", help="show job status")
    job_status.add_argument("--job", required=True, type=Path)
    job_status.add_argument("--json", action="store_true")
    job_status.set_defaults(func=cmd_job_status)

    job_list = job_sub.add_parser("list", help="list tracked jobs")
    job_list.add_argument("--root", default=Path("runs/jobs"), type=Path)
    job_list.set_defaults(func=cmd_job_list)

    job_logs = job_sub.add_parser("logs", help="show job logs")
    job_logs.add_argument("--job", required=True, type=Path)
    job_logs.add_argument("--stream", default="stdout", choices=["stdout", "stderr", "events"])
    job_logs.add_argument("--tail", default=40, type=int)
    job_logs.set_defaults(func=cmd_job_logs)

    job_collect = job_sub.add_parser("collect", help="run collect commands for a tracked job")
    job_collect.add_argument("--job", required=True, type=Path)
    job_collect.add_argument("--timeout-seconds", required=False, type=int)
    job_collect.set_defaults(func=cmd_job_collect)

    workbench = sub.add_parser("workbench", help="static workbench export commands")
    workbench_sub = workbench.add_subparsers(required=True)
    workbench_build = workbench_sub.add_parser("build", help="write a static HTML workbench")
    workbench_build.add_argument("--root", default=Path("."), type=Path)
    workbench_build.add_argument("--registry-root", required=False, type=Path)
    workbench_build.add_argument("--rounds-root", required=False, type=Path)
    workbench_build.add_argument("--train-runs-root", required=False, type=Path)
    workbench_build.add_argument("--eval-runs-root", required=False, type=Path)
    workbench_build.add_argument("--jobs-root", required=False, type=Path)
    workbench_build.add_argument("--output", default=DEFAULT_WORKBENCH_OUTPUT, type=Path)
    workbench_build.add_argument("--title", default="BadcaseFlow Workbench")
    workbench_build.set_defaults(func=cmd_workbench_build)

    remote = sub.add_parser("remote", help="remote server and AutoDL planning commands")
    remote_sub = remote.add_subparsers(required=True)
    remote_add = remote_sub.add_parser("add", help="register a remote target")
    remote_add.add_argument("--name", required=True)
    remote_add.add_argument("--host", required=True)
    remote_add.add_argument("--workdir", required=True)
    remote_add.add_argument("--port", default=22, type=int)
    remote_add.add_argument("--python", default="python")
    remote_add.add_argument("--config", default=Path(".badcaseflow/remotes.json"), type=Path)
    remote_add.set_defaults(func=cmd_remote_add)

    remote_list = remote_sub.add_parser("list", help="list remote targets")
    remote_list.add_argument("--config", default=Path(".badcaseflow/remotes.json"), type=Path)
    remote_list.set_defaults(func=cmd_remote_list)

    remote_plan = remote_sub.add_parser("plan", help="write a remote dry-run plan")
    remote_plan.add_argument("--target", required=True)
    remote_plan.add_argument("--run", required=True, type=Path)
    remote_plan.add_argument("--recipe", required=True, type=Path)
    remote_plan.add_argument("--workspace", required=False)
    remote_plan.add_argument("--config", default=Path(".badcaseflow/remotes.json"), type=Path)
    remote_plan.add_argument("--output", required=False, type=Path)
    remote_plan.set_defaults(func=cmd_remote_plan)

    demo = sub.add_parser("demo", help="run the complete local demo flow")
    demo.add_argument("--workspace", default="demo-agent")
    demo.add_argument("--run", default=Path("runs/demo"), type=Path)
    demo.add_argument("--seed", default=Path("examples/datasets/seed_tasks.jsonl"), type=Path)
    demo.add_argument("--suite", default=Path("examples/datasets/eval_tasks.jsonl"), type=Path)
    demo.add_argument("--agent", default="mock", choices=["mock", "mock-buggy"])
    demo.add_argument("--sft-recipe", default=Path("examples/recipes/sft_llamafactory.example.yaml"), type=Path)
    demo.set_defaults(func=cmd_demo)

    return parser


def cmd_init_workspace(args: argparse.Namespace) -> int:
    workspace = init_workspace(args.root, args.workspace, force=args.force)
    print(f"workspace={workspace['workspace_id']} config={args.root / '.badcaseflow' / 'workspace.json'}")
    return 0


def cmd_version(args: argparse.Namespace) -> int:
    payload = {"name": "badcaseflow", "version": __version__}
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"badcaseflow={__version__}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    report = _doctor_report(args.root)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report["status"] == "ok" or not args.strict else 2

    print(f"badcaseflow={report['badcaseflow_version']}")
    print(f"python={report['python']['version']}")
    print(f"python_ok={report['python']['ok']}")
    print(f"root={report['root']}")
    print(f"workspace={report['workspace'].get('workspace_id') or 'not-initialized'}")
    print(f"training_adapters={report['training_adapters']}")
    print(f"eval_adapters={report['eval_adapters']}")
    for check in report["paths"]:
        print(f"{check['path']}={check['status']}")
    if report["missing"]:
        print("missing=" + ",".join(report["missing"]))
    print(f"status={report['status']}")
    return 0 if report["status"] == "ok" or not args.strict else 2


def _doctor_report(root: Path) -> dict[str, Any]:
    root = root.resolve()
    python_version = sys.version.split()[0]
    python_ok = sys.version_info >= (3, 11)
    try:
        workspace = load_workspace(root)
    except FileNotFoundError:
        workspace = {}
    required_paths = (
        "examples/datasets/seed_tasks.jsonl",
        "examples/datasets/eval_tasks.jsonl",
        "examples/datasets/external_traces.jsonl",
        "examples/datasets/gsm8k_tiny.jsonl",
        "examples/recipes/sft_llamafactory.example.yaml",
        "examples/recipes/dpo_llamafactory.example.yaml",
        "examples/recipes/kto_llamafactory.example.yaml",
        "examples/recipes/sft_verl.example.yaml",
        "examples/recipes/grpo_verl.example.yaml",
        "examples/recipes/gsm8k_sft_llamafactory.example.yaml",
        "examples/recipes/gsm8k_grpo_verl.example.yaml",
        "examples/recipes/gsm8k_ppo_verl_autodl.example.yaml",
        "examples/recipes/opd_verl.example.yaml",
        "examples/recipes/command_smoke.example.yaml",
        "examples/evals/builtin_rule.example.yaml",
        "examples/evals/command_eval_smoke.example.yaml",
        "examples/evals/promptfoo.example.yaml",
        "examples/evals/opencompass.example.yaml",
        "examples/evals/lighteval.example.yaml",
        "CHANGELOG.md",
        "CONTRIBUTING_zh.md",
        "LICENSE",
        "README.md",
        "docs/autodl_gsm8k_quickstart_zh.md",
        "docs/github_release_checklist_zh.md",
        "docs/first_version_scope_zh.md",
        "docs/product_positioning_zh.md",
        "docs/technical_design_zh.md",
        "examples/README_zh.md",
        ".github/workflows/ci.yml",
        "pyproject.toml",
    )
    path_checks = [
        {
            "path": relative,
            "status": "ok" if (root / relative).exists() else "missing",
        }
        for relative in required_paths
    ]
    missing = [check["path"] for check in path_checks if check["status"] != "ok"]
    status = "ok" if python_ok and not missing else "warning"
    return {
        "status": status,
        "badcaseflow_version": __version__,
        "python": {"version": python_version, "ok": python_ok},
        "root": str(root),
        "workspace": workspace,
        "training_adapters": len(list_training_adapters()),
        "eval_adapters": len(list_eval_adapters()),
        "paths": path_checks,
        "missing": missing,
    }


def cmd_ingest(args: argparse.Namespace) -> int:
    records = read_jsonl(args.input)
    accepted: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for record in records:
        validation_errors = validate_task_sample(record)
        if record.get("workspace_id") != args.workspace:
            validation_errors.append("workspace_id does not match --workspace")
        if validation_errors:
            errors.append({"record": record, "errors": validation_errors})
        else:
            accepted.append(record)

    store = LocalRunStore(args.run)
    store.init(args.workspace, str(args.input))
    store.save_records("tasks", "tasks.jsonl", accepted)
    store.save_records("ingest_errors", "ingest_errors.jsonl", errors)
    store.save_report(
        "ingest_summary",
        "ingest_summary.json",
        {"input": str(args.input), "accepted": len(accepted), "errors": len(errors)},
    )
    print(f"ingested={len(accepted)} errors={len(errors)} run={args.run}")
    return 0 if not errors else 2


def cmd_import_traces(args: argparse.Namespace) -> int:
    records = read_jsonl(args.input)
    accepted: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for record in records:
        validation_errors = validate_agent_trace(record)
        if record.get("workspace_id") != args.workspace:
            validation_errors.append("workspace_id does not match --workspace")
        if validation_errors:
            errors.append({"record": record, "errors": validation_errors})
        else:
            accepted.append(record)

    store = LocalRunStore(args.run)
    if store.manifest_path.exists():
        _require_workspace(store, args.workspace)
    else:
        store.init(args.workspace, str(args.input))
    store.save_records("traces", "traces.jsonl", accepted)
    store.save_records("trace_import_errors", "trace_import_errors.jsonl", errors)
    store.save_report(
        "trace_import_summary",
        "trace_import_summary.json",
        {"input": str(args.input), "accepted": len(accepted), "errors": len(errors)},
    )
    print(f"traces={len(accepted)} errors={len(errors)} run={args.run}")
    return 0 if not errors else 2


def cmd_inspect(args: argparse.Namespace) -> int:
    store = LocalRunStore(args.run)
    manifest = store.load_manifest()
    artifacts = manifest.get("artifacts", {})
    print(f"run_id={manifest.get('run_id')} workspace={manifest.get('workspace_id')}")
    for key, filename in sorted(artifacts.items()):
        path = args.run / filename
        count = _count_jsonl(path) if filename.endswith(".jsonl") and path.exists() else "-"
        print(f"{key}: {filename} count={count}")
    return 0


def cmd_rollout(args: argparse.Namespace) -> int:
    store = LocalRunStore(args.run)
    _require_workspace(store, args.workspace)
    tasks = store.load_records("tasks.jsonl")
    traces = rollout_tasks(tasks, agent=args.agent)
    store.save_records("traces", "traces.jsonl", traces)
    store.save_report("rollout_summary", "rollout_summary.json", {"agent": args.agent, "traces": len(traces)})
    print(f"traces={len(traces)} output={args.run / 'traces.jsonl'}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    store = LocalRunStore(args.run)
    _require_workspace(store, args.workspace)
    traces = store.load_records("traces.jsonl")
    suite = read_jsonl(args.suite)
    results, report, accepted, rejected = evaluate_traces(traces, suite)
    store.save_records("eval_results", "eval_results.jsonl", results)
    store.save_report("eval_report", "eval_report.json", report)
    store.save_records("accepted", "accepted.jsonl", accepted)
    store.save_records("rejected", "rejected.jsonl", rejected)
    print(f"total={report['total']} passed={report['passed']} failed={report['failed']} pass_rate={report['pass_rate']}")
    return 0 if report["failed"] == 0 else 2


def cmd_evals_adapters(args: argparse.Namespace) -> int:
    print("backend\tstage\t名称\t别名\t说明")
    for row in list_eval_adapters():
        aliases = ",".join(row["aliases"]) if row["aliases"] else "-"
        print(f"{row['backend']}\t{row['stage']}\t{row['title']}\t{aliases}\t{row['description']}")
    return 0


def cmd_evals_dry_run(args: argparse.Namespace) -> int:
    manifest = build_eval_dry_run(args.recipe, workspace=args.workspace, registry_root=args.registry_root)
    path = save_dry_run_manifest(manifest, args.output_root)
    print(f"dry_run={path}")
    print(f"backend={manifest['backend']} stage={manifest['stage']}")
    print(f"preflight={manifest['preflight']['status']}")
    print(f"datasets={len((manifest.get('dataset_resolution') or {}).get('datasets', []))}")
    execution = manifest.get("execution") or {}
    if execution.get("command"):
        print("command=" + " ".join(str(part) for part in execution["command"]))
    else:
        print(f"execution={execution.get('type')}")
    return 0 if manifest["preflight"]["status"] != "failed" else 2


def cmd_evals_run(args: argparse.Namespace) -> int:
    manifest = build_eval_dry_run(args.recipe, workspace=args.workspace, registry_root=args.registry_root)
    preflight = manifest["preflight"]
    if preflight["status"] == "failed" and not args.allow_failed_preflight:
        print(f"preflight={preflight['status']}")
        for error in preflight.get("errors", []):
            print(f"error={error}")
        return 2

    run_id = args.run_id or create_eval_run_id(manifest)
    run_dir = args.output_root / "eval_runs" / run_id
    if args.plan_only:
        result = plan_eval_run(manifest, run_dir)
    else:
        result = run_eval_manifest(
            manifest,
            run_dir,
            cwd=args.cwd,
            env=parse_env_overrides(args.env),
            timeout_seconds=args.timeout_seconds,
        )

    print(f"eval_run={run_dir}")
    print(f"status={result['status']} quality_status={result.get('quality_status')}")
    print(f"backend={manifest['backend']} stage={manifest['stage']}")
    if result.get("eval_report"):
        report = result["eval_report"]
        print(
            f"total={report.get('total')} passed={report.get('passed')} "
            f"failed={report.get('failed')} pass_rate={report.get('pass_rate')}"
        )
    blocked = result.get("quality_status") == "blocked"
    return 2 if result["status"] not in {"planned", "succeeded"} or (args.fail_on_blocked and blocked) else 0


def cmd_evals_inspect(args: argparse.Namespace) -> int:
    manifest_path = args.run / "manifest.json"
    status_path = args.run / "status.json"
    metrics_path = args.run / "metrics.jsonl"
    events_path = args.run / "events.jsonl"
    status = read_json(status_path if status_path.exists() else manifest_path)
    print(f"eval_run_id={status.get('eval_run_id')} status={status.get('status')}")
    print(
        f"backend={status.get('backend')} stage={status.get('stage')} "
        f"quality_status={status.get('quality_status')} return_code={status.get('return_code')}"
    )
    report = status.get("eval_report") or {}
    if report:
        print(
            f"total={report.get('total')} passed={report.get('passed')} "
            f"failed={report.get('failed')} pass_rate={report.get('pass_rate')}"
        )
    metrics = read_jsonl(metrics_path) if metrics_path.exists() else []
    print(f"metrics={len(metrics)} events={_count_jsonl(events_path) if events_path.exists() else 0}")
    for record in metrics[-max(args.tail, 0) :]:
        values = record.get("values", {})
        rendered = " ".join(f"{key}={value}" for key, value in sorted(values.items()))
        print(f"metric step={record.get('step')} {rendered}".rstrip())
    return 0 if status.get("status") in {"planned", "succeeded"} else 2


def cmd_evals_list(args: argparse.Namespace) -> int:
    print("eval_run_id\tstatus\tquality_status\tbackend\tstage\trecipe_id\tupdated_at")
    if not args.root.exists():
        return 0
    rows: list[dict[str, Any]] = []
    for child in args.root.iterdir():
        if not child.is_dir():
            continue
        status_path = child / "status.json"
        manifest_path = child / "manifest.json"
        if not status_path.exists() and not manifest_path.exists():
            continue
        rows.append(read_json(status_path if status_path.exists() else manifest_path))
    for row in sorted(rows, key=lambda item: str(item.get("updated_at", "")), reverse=True):
        print(
            "\t".join(
                _cell(value)
                for value in (
                    row.get("eval_run_id"),
                    row.get("status"),
                    row.get("quality_status"),
                    row.get("backend"),
                    row.get("stage"),
                    row.get("recipe_id"),
                    row.get("updated_at"),
                )
            )
        )
    return 0


def cmd_analyze_cases(args: argparse.Namespace) -> int:
    store = LocalRunStore(args.run)
    _require_workspace(store, args.workspace)
    eval_results = store.load_records("eval_results.jsonl")
    traces = store.load_records("traces.jsonl")
    report = analyze_cases(eval_results, traces)
    store.save_report("case_report", "case_report.json", report)
    markdown_path = store.path("case_report.md")
    markdown_path.write_text(render_case_report_markdown(report), encoding="utf-8", newline="\n")
    store.update_artifact("case_report_md", "case_report.md")
    print(f"failed_cases={report['total_failed_cases']} actions={report['recommended_action_counts']}")
    return 0


def cmd_export_sft(args: argparse.Namespace) -> int:
    store = LocalRunStore(args.run)
    _require_workspace(store, args.workspace)
    source = "accepted.jsonl" if (args.run / "accepted.jsonl").exists() else "traces.jsonl"
    traces = store.load_records(source)
    records = export_sft(traces)
    write_jsonl(args.output, records)
    store.update_artifact("sft", _relative_to_run(args.output, args.run))
    print(f"sft_records={len(records)} output={args.output}")
    return 0


def cmd_export_preference(args: argparse.Namespace) -> int:
    store = LocalRunStore(args.run)
    _require_workspace(store, args.workspace)
    report = read_json(args.run / "case_report.json")
    records = export_preference(report.get("cases", []))
    write_jsonl(args.output, records)
    store.update_artifact("preference", _relative_to_run(args.output, args.run))
    print(f"preference_records={len(records)} output={args.output}")
    return 0


def cmd_export_rl_prompt(args: argparse.Namespace) -> int:
    store = LocalRunStore(args.run)
    _require_workspace(store, args.workspace)
    tasks = store.load_records("tasks.jsonl")
    records = export_rl_prompts(tasks)
    write_jsonl(args.output, records)
    store.update_artifact("rl_prompts", _relative_to_run(args.output, args.run))
    print(f"rl_prompt_records={len(records)} output={args.output}")
    return 0


def cmd_train_dry_run(args: argparse.Namespace) -> int:
    manifest = build_train_dry_run(args.recipe, workspace=args.workspace, registry_root=args.registry_root)
    path = save_dry_run_manifest(manifest, args.output_root)
    print(f"dry_run={path}")
    print(f"backend={manifest['backend']} stage={manifest['stage']}")
    print(f"preflight={manifest['preflight']['status']}")
    print(f"datasets={len((manifest.get('dataset_resolution') or {}).get('datasets', []))}")
    print("command=" + " ".join(str(part) for part in manifest["command"]))
    return 0 if manifest["preflight"]["status"] != "failed" else 2


def cmd_train_run(args: argparse.Namespace) -> int:
    manifest = build_train_dry_run(args.recipe, workspace=args.workspace, registry_root=args.registry_root)
    preflight = manifest["preflight"]
    if preflight["status"] == "failed" and not args.allow_failed_preflight:
        print(f"preflight={preflight['status']}")
        for error in preflight.get("errors", []):
            print(f"error={error}")
        return 2

    run_id = args.run_id or create_train_run_id(manifest)
    run_dir = args.output_root / "train_runs" / run_id
    if args.plan_only:
        result = plan_train_run(manifest, run_dir)
    else:
        result = run_train_manifest(
            manifest,
            run_dir,
            cwd=args.cwd,
            env=parse_env_overrides(args.env),
            timeout_seconds=args.timeout_seconds,
        )

    print(f"train_run={run_dir}")
    print(f"status={result['status']}")
    print(f"backend={manifest['backend']} stage={manifest['stage']}")
    print("command=" + " ".join(str(part) for part in manifest["command"]))
    return 0 if result["status"] in {"planned", "succeeded"} else 2


def cmd_train_inspect(args: argparse.Namespace) -> int:
    manifest_path = args.run / "manifest.json"
    status_path = args.run / "status.json"
    metrics_path = args.run / "metrics.jsonl"
    events_path = args.run / "events.jsonl"
    status = read_json(status_path if status_path.exists() else manifest_path)
    print(f"train_run_id={status.get('train_run_id')} status={status.get('status')}")
    print(f"backend={status.get('backend')} stage={status.get('stage')} return_code={status.get('return_code')}")
    print(f"duration_seconds={status.get('duration_seconds')} recipe_id={status.get('recipe_id')}")
    command = status.get("command") or []
    print("command=" + " ".join(str(part) for part in command))
    diagnosis = status.get("diagnosis") or {}
    if diagnosis:
        print(f"diagnosis={diagnosis.get('category')} severity={diagnosis.get('severity')}")
        print(f"summary={diagnosis.get('summary')}")

    metrics = read_jsonl(metrics_path) if metrics_path.exists() else []
    print(f"metrics={len(metrics)} events={_count_jsonl(events_path) if events_path.exists() else 0}")
    for record in metrics[-max(args.tail, 0) :]:
        values = record.get("values", {})
        rendered = " ".join(f"{key}={value}" for key, value in sorted(values.items()))
        print(f"metric step={record.get('step')} {rendered}".rstrip())
    return 0 if status.get("status") in {"planned", "succeeded"} else 2


def cmd_train_list(args: argparse.Namespace) -> int:
    print("train_run_id\tstatus\tbackend\tstage\trecipe_id\tupdated_at")
    if not args.root.exists():
        return 0
    rows: list[dict[str, Any]] = []
    for child in args.root.iterdir():
        if not child.is_dir():
            continue
        status_path = child / "status.json"
        manifest_path = child / "manifest.json"
        if not status_path.exists() and not manifest_path.exists():
            continue
        status = read_json(status_path if status_path.exists() else manifest_path)
        rows.append(status)
    for row in sorted(rows, key=lambda item: str(item.get("updated_at", "")), reverse=True):
        print(
            "\t".join(
                str(value or "-")
                for value in (
                    row.get("train_run_id"),
                    row.get("status"),
                    row.get("backend"),
                    row.get("stage"),
                    row.get("recipe_id"),
                    row.get("updated_at"),
                )
            )
        )
    return 0


def cmd_train_adapters(args: argparse.Namespace) -> int:
    print("backend\tstage\t名称\t别名\t说明")
    for row in list_training_adapters():
        aliases = ",".join(row["aliases"]) if row["aliases"] else "-"
        print(f"{row['backend']}\t{row['stage']}\t{row['title']}\t{aliases}\t{row['description']}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    before = read_json(args.before)
    after = read_json(args.after)
    delta = round(float(after.get("pass_rate", 0.0)) - float(before.get("pass_rate", 0.0)), 4)
    print(f"before_pass_rate={before.get('pass_rate')} after_pass_rate={after.get('pass_rate')} delta={delta}")
    return 0


def cmd_promote_dry_run(args: argparse.Namespace) -> int:
    report = read_json(args.eval_report)
    decision = decide_promotion(
        report,
        min_pass_rate=args.min_pass_rate,
        allow_failed_cases=args.allow_failed_cases,
    )
    print(
        " ".join(
            [
                f"model={args.model}",
                f"decision={decision['decision']}",
                f"pass_rate={decision['pass_rate']}",
                f"failed={decision['failed']}",
                f"min_pass_rate={decision['min_pass_rate']}",
            ]
        )
    )
    if decision["reasons"]:
        print("reasons=" + ",".join(decision["reasons"]))
    return 0 if decision["decision"] == "approved" else 2


def cmd_data_prepare_gsm8k(args: argparse.Namespace) -> int:
    summary = prepare_gsm8k(
        args.input,
        args.output_dir,
        val_input_path=args.val_input,
        workspace_id=args.workspace,
        source_name=args.source_name,
        suite_id=args.suite_id,
        train_ratio=args.train_ratio,
        limit=args.limit,
        val_limit=args.val_limit,
        write_parquet=args.parquet,
    )
    counts = summary["counts"]
    print(f"prepared_gsm8k={args.output_dir}")
    print(
        " ".join(
            [
                f"train={counts['train']}",
                f"val={counts['val']}",
                f"tasks={counts['task_samples']}",
                f"eval_items={counts['eval_items']}",
            ]
        )
    )
    for key in (
        "task_samples",
        "eval_suite",
        "sft_train",
        "sft_val",
        "rl_train",
        "rl_val",
        "verl_train_jsonl",
        "verl_val_jsonl",
        "dataset_info",
        "manifest",
    ):
        print(f"{key}={summary['outputs'][key]}")
    parquet = summary.get("parquet") or {}
    if parquet.get("written"):
        for key, path in sorted((parquet.get("paths") or {}).items()):
            print(f"{key}={path}")
    else:
        print(f"parquet={'requested' if parquet.get('requested') else 'not_requested'}")

    if args.register:
        registered = _register_prepared_gsm8k_outputs(args.registry_root, args.workspace, summary)
        print(f"registry={registry_path(args.registry_root)}")
        for manifest in registered:
            print(
                " ".join(
                    [
                        f"dataset_version_id={manifest['dataset_version_id']}",
                        f"dataset_id={manifest['dataset_id']}",
                        f"kind={manifest['kind']}",
                        f"records={manifest['record_count']}",
                        f"schema_status={manifest['schema_status']}",
                    ]
                )
            )
    return 0


def _register_prepared_gsm8k_outputs(
    registry_root: Path,
    workspace: str,
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    outputs = summary.get("outputs") or {}
    source = {
        "builder": "prepare-gsm8k",
        "source_name": summary.get("source_name"),
        "input_path": summary.get("input_path"),
        "val_input_path": summary.get("val_input_path"),
    }
    specs = [
        ("task_samples", "task", "gsm8k-task-samples"),
        ("eval_suite", "eval_suite", "gsm8k-eval-suite"),
        ("sft_train", "sft", "gsm8k-sft-train"),
        ("sft_val", "sft", "gsm8k-sft-val"),
        ("rl_train", "rl_prompt", "gsm8k-rl-train"),
        ("rl_val", "rl_prompt", "gsm8k-rl-val"),
        ("verl_train_jsonl", "rl_prompt", "gsm8k-verl-train"),
        ("verl_val_jsonl", "rl_prompt", "gsm8k-verl-val"),
    ]
    parquet_paths = (summary.get("parquet") or {}).get("paths") or {}
    for key, dataset_id in (
        ("verl_train_parquet", "gsm8k-verl-train-parquet"),
        ("verl_val_parquet", "gsm8k-verl-val-parquet"),
    ):
        if key in parquet_paths:
            specs.append((key, "rl_prompt", dataset_id))

    # 同一批生成结果分别登记，后续 recipe 可以按 dataset_id 做解析和血缘追踪。
    registered: list[dict[str, Any]] = []
    for key, kind, dataset_id in specs:
        raw_path = outputs.get(key) or parquet_paths.get(key)
        if not raw_path:
            continue
        registered.append(
            register_dataset(
                registry_root,
                path=Path(str(raw_path)),
                kind=kind,
                dataset_id=dataset_id,
                workspace_id=workspace,
                description=f"GSM8K {key.replace('_', ' ')}",
                source=source,
            )
        )
    return registered


def cmd_dataset_inspect(args: argparse.Namespace) -> int:
    manifest = build_dataset_version(
        args.path,
        kind=args.kind,
        dataset_id=args.dataset_id,
        workspace_id=args.workspace,
        description=args.description,
    )
    if args.output:
        write_dataset_manifest(args.output, manifest)
        print(f"manifest={args.output}")
    if args.json:
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(
            " ".join(
                [
                    f"dataset_version_id={manifest['dataset_version_id']}",
                    f"kind={manifest['kind']}",
                    f"records={manifest['record_count']}",
                    f"schema_status={manifest['schema_status']}",
                    f"bytes={manifest['bytes']}",
                ]
            )
        )
        print(f"sha256={manifest['sha256']}")
        if manifest.get("schema_errors"):
            print(f"schema_errors={len(manifest['schema_errors'])}")
        if manifest.get("parse_errors"):
            print(f"parse_errors={len(manifest['parse_errors'])}")
    return 0 if manifest["schema_status"] == "passed" else 2


def cmd_dataset_register(args: argparse.Namespace) -> int:
    manifest = register_dataset(
        args.root,
        path=args.path,
        kind=args.kind,
        dataset_id=args.dataset_id,
        workspace_id=args.workspace,
        description=args.description,
    )
    print(f"registry={registry_path(args.root)}")
    print(
        " ".join(
            [
                f"dataset_version_id={manifest['dataset_version_id']}",
                f"dataset_id={manifest['dataset_id']}",
                f"kind={manifest['kind']}",
                f"records={manifest['record_count']}",
                f"schema_status={manifest['schema_status']}",
            ]
        )
    )
    return 0 if manifest["schema_status"] == "passed" else 2


def cmd_dataset_list(args: argparse.Namespace) -> int:
    print("dataset_version_id\tdataset_id\tkind\trecords\tschema_status\tpath")
    for dataset in list_datasets(args.root):
        print(
            "\t".join(
                _cell(value)
                for value in (
                    dataset.get("dataset_version_id"),
                    dataset.get("dataset_id"),
                    dataset.get("kind"),
                    dataset.get("record_count"),
                    dataset.get("schema_status"),
                    dataset.get("path"),
                )
            )
        )
    return 0


def cmd_dataset_show(args: argparse.Namespace) -> int:
    dataset = get_dataset(args.root, args.dataset_version_id)
    print(json.dumps(dataset, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def cmd_registry_index_train_run(args: argparse.Namespace) -> int:
    result = index_train_run(args.root, args.run)
    print(f"registry={result['registry']}")
    print(f"train_run_id={result['train_run_id']} indexed={result['indexed']}")
    return 0


def cmd_registry_index_eval_run(args: argparse.Namespace) -> int:
    result = index_eval_run(args.root, args.run)
    print(f"registry={result['registry']}")
    print(f"eval_run_id={result['eval_run_id']} indexed={result['indexed']}")
    return 0


def cmd_registry_artifacts(args: argparse.Namespace) -> int:
    print("artifact_id\tkind\tstatus\tbackend\tstage\tpath")
    for artifact in list_artifacts(args.root):
        print(
            "\t".join(
                str(value or "-")
                for value in (
                    artifact.get("artifact_id"),
                    artifact.get("kind"),
                    artifact.get("status"),
                    artifact.get("backend"),
                    artifact.get("stage"),
                    artifact.get("path"),
                )
            )
        )
    return 0


def cmd_registry_register_model(args: argparse.Namespace) -> int:
    index_train_run(args.root, args.run)
    model = register_model(
        args.root,
        model_id=args.model_id,
        run_dir=args.run,
        model_path=args.model_path,
        description=args.description,
        allow_non_succeeded=args.allow_non_succeeded,
    )
    print(f"registry={registry_path(args.root)}")
    print(f"model_id={model['model_id']} status={model['status']} train_run_id={model['train_run_id']}")
    print(f"model_path={model['model_path']}")
    return 0


def cmd_registry_models(args: argparse.Namespace) -> int:
    print("model_id\tstatus\tbackend\tstage\ttrain_run_id\tmodel_path")
    for model in list_models(args.root):
        print(
            "\t".join(
                str(value or "-")
                for value in (
                    model.get("model_id"),
                    model.get("status"),
                    model.get("backend"),
                    model.get("stage"),
                    model.get("train_run_id"),
                    model.get("model_path"),
                )
            )
        )
    return 0


def cmd_registry_show_model(args: argparse.Namespace) -> int:
    model = get_model(args.root, args.model_id)
    print(json.dumps(model, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def cmd_registry_attach_eval(args: argparse.Namespace) -> int:
    result = attach_eval_report(
        args.root,
        model_id=args.model_id,
        eval_report_path=args.eval_report,
        min_pass_rate=args.min_pass_rate,
        allow_failed_cases=args.allow_failed_cases,
    )
    evaluation = result["evaluation"]
    promotion = result["promotion"]
    decision = result["decision"]
    print(f"registry={registry_path(args.root)}")
    print(
        " ".join(
            [
                f"model_id={args.model_id}",
                f"evaluation_id={evaluation['evaluation_id']}",
                f"promotion_id={promotion['promotion_id']}",
                f"decision={decision['decision']}",
                f"pass_rate={decision['pass_rate']}",
                f"failed={decision['failed']}",
            ]
        )
    )
    if decision["reasons"]:
        print("reasons=" + ",".join(decision["reasons"]))
    return 2 if args.fail_on_blocked and decision["decision"] != "approved" else 0


def cmd_registry_attach_eval_run(args: argparse.Namespace) -> int:
    result = attach_eval_run(
        args.root,
        model_id=args.model_id,
        eval_run_dir=args.eval_run,
        min_pass_rate=args.min_pass_rate,
        allow_failed_cases=args.allow_failed_cases,
    )
    evaluation = result["evaluation"]
    promotion = result["promotion"]
    decision = result["decision"]
    print(f"registry={registry_path(args.root)}")
    print(
        " ".join(
            [
                f"model_id={args.model_id}",
                f"eval_run_id={evaluation.get('eval_run_id')}",
                f"evaluation_id={evaluation['evaluation_id']}",
                f"promotion_id={promotion['promotion_id']}",
                f"decision={decision['decision']}",
                f"pass_rate={decision['pass_rate']}",
                f"failed={decision['failed']}",
            ]
        )
    )
    if decision["reasons"]:
        print("reasons=" + ",".join(decision["reasons"]))
    return 2 if args.fail_on_blocked and decision["decision"] != "approved" else 0


def cmd_registry_evaluations(args: argparse.Namespace) -> int:
    print("evaluation_id\tmodel_id\teval_run_id\tdecision\tpass_rate\tfailed\treport_path")
    for evaluation in list_evaluations(args.root):
        summary = evaluation.get("summary") or {}
        print(
            "\t".join(
                _cell(value)
                for value in (
                    evaluation.get("evaluation_id"),
                    evaluation.get("model_id"),
                    evaluation.get("eval_run_id"),
                    evaluation.get("decision"),
                    summary.get("pass_rate"),
                    summary.get("failed"),
                    evaluation.get("report_path"),
                )
            )
        )
    return 0


def cmd_registry_promotions(args: argparse.Namespace) -> int:
    print("promotion_id\tmodel_id\tdecision\tpass_rate\tmin_pass_rate\tfailed\treasons")
    for promotion in list_promotions(args.root):
        print(
            "\t".join(
                _cell(value)
                for value in (
                    promotion.get("promotion_id"),
                    promotion.get("model_id"),
                    promotion.get("decision"),
                    promotion.get("pass_rate"),
                    promotion.get("min_pass_rate"),
                    promotion.get("failed"),
                    ",".join(promotion.get("reasons") or []),
                )
            )
        )
    return 0


def cmd_registry_model_lineage(args: argparse.Namespace) -> int:
    lineage = get_model_lineage(args.root, args.model_id)
    print(json.dumps(lineage, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def cmd_deploy_plan(args: argparse.Namespace) -> int:
    plan = build_deployment_plan(
        args.root,
        model_id=args.model_id,
        backend=args.backend,
        mode=args.mode,
        deployment_id=args.deployment_id,
        served_model_name=args.served_model_name,
        host=args.host,
        port=args.port,
        python=args.python,
        endpoint=args.endpoint,
        canary_percent=args.canary_percent,
        min_pass_rate=args.min_pass_rate,
        previous_model_id=args.previous_model_id,
        rollback_endpoint=args.rollback_endpoint,
        allow_blocked=args.allow_blocked,
    )
    output = args.output or Path("runs") / "deployments" / f"{plan['deployment_id']}.json"
    write_json(output, plan)
    script_path = output.with_suffix(".ps1")
    script_path.write_text(render_deployment_script(plan), encoding="utf-8", newline="\n")
    if args.register:
        register_deployment(args.root, plan)
    print(f"deployment_plan={output}")
    print(f"script={script_path}")
    print(
        " ".join(
            [
                f"deployment_id={plan['deployment_id']}",
                f"model_id={plan['model_id']}",
                f"backend={plan['backend']}",
                f"status={plan['status']}",
                f"registered={bool(args.register)}",
            ]
        )
    )
    serve_commands = plan.get("commands", {}).get("serve", [])
    if serve_commands:
        print("next=" + str(serve_commands[0]))
    return 0


def cmd_deploy_list(args: argparse.Namespace) -> int:
    print("deployment_id\tmodel_id\tbackend\tstatus\tendpoint\tupdated_at")
    for deployment in list_deployments(args.root):
        service = deployment.get("service") if isinstance(deployment.get("service"), dict) else {}
        print(
            "\t".join(
                _cell(value)
                for value in (
                    deployment.get("deployment_id"),
                    deployment.get("model_id"),
                    deployment.get("backend"),
                    deployment.get("status"),
                    service.get("endpoint"),
                    deployment.get("updated_at"),
                )
            )
        )
    return 0


def cmd_deploy_show(args: argparse.Namespace) -> int:
    deployment = get_deployment(args.root, args.deployment_id)
    print(json.dumps(deployment, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def cmd_round_run_local(args: argparse.Namespace) -> int:
    manifest = run_local_round(
        workspace_id=args.workspace,
        round_dir=args.round,
        seed_path=args.seed,
        suite_path=args.suite,
        agent=args.agent,
        recipe_path=args.recipe,
        train_mode=args.train_mode,
        registry_root=args.registry_root,
        model_id=args.model_id,
        model_path=args.model_path,
        train_run_id=args.run_id,
        cwd=args.cwd,
        env=parse_env_overrides(args.env),
        timeout_seconds=args.timeout_seconds,
        min_pass_rate=args.min_pass_rate,
        allow_failed_cases=args.allow_failed_cases,
        allow_non_succeeded_model=args.allow_non_succeeded_model,
        attach_round_eval=args.attach_round_eval,
        candidate_eval_report=args.candidate_eval_report,
        candidate_eval_recipe_path=args.candidate_eval_recipe,
        candidate_eval_mode=args.candidate_eval_mode,
        candidate_eval_run_id=args.candidate_eval_run_id,
    )
    print(f"round={args.round}")
    print(f"status={manifest.get('status')} quality_status={manifest.get('quality_status')}")
    summary = manifest.get("summary") or {}
    print(
        " ".join(
            [
                f"tasks={summary.get('tasks')}",
                f"traces={summary.get('traces')}",
                f"pass_rate={summary.get('pass_rate')}",
                f"failed={summary.get('failed')}",
            ]
        )
    )
    if summary.get("model_id"):
        print(f"model_id={summary.get('model_id')} promotion_decision={summary.get('promotion_decision')}")
    return 2 if args.fail_on_blocked and manifest.get("quality_status") in {"blocked", "needs_iteration"} else 0


def cmd_round_list(args: argparse.Namespace) -> int:
    print("round_id\tstatus\tquality_status\tworkspace\tpass_rate\tfailed\tupdated_at")
    for item in list_rounds(args.root):
        summary = item.get("summary") or {}
        print(
            "\t".join(
                _cell(value)
                for value in (
                    item.get("round_id"),
                    item.get("status"),
                    item.get("quality_status"),
                    item.get("workspace_id"),
                    summary.get("pass_rate"),
                    summary.get("failed"),
                    item.get("updated_at"),
                )
            )
        )
    return 0


def cmd_round_inspect(args: argparse.Namespace) -> int:
    manifest = load_round(args.round)
    if args.json:
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    print(f"round_id={manifest.get('round_id')} status={manifest.get('status')}")
    print(f"workspace={manifest.get('workspace_id')} quality_status={manifest.get('quality_status')}")
    summary = manifest.get("summary") or {}
    print(
        " ".join(
            [
                f"tasks={summary.get('tasks')}",
                f"traces={summary.get('traces')}",
                f"pass_rate={summary.get('pass_rate')}",
                f"failed={summary.get('failed')}",
            ]
        )
    )
    if summary.get("model_id"):
        print(f"model_id={summary.get('model_id')} promotion_decision={summary.get('promotion_decision')}")
    for stage in manifest.get("stages", []):
        print(f"stage={stage.get('name')} status={stage.get('status')}")
    return 0 if manifest.get("status") == "succeeded" else 2


def cmd_round_plan_next(args: argparse.Namespace) -> int:
    plan = build_iteration_plan_from_round(args.round)
    outputs = write_iteration_plan(args.round, plan, output=args.output)
    print(f"iteration_plan={args.round / outputs['iteration_plan'] if args.output is None else args.output}")
    print(f"quality_status={plan.get('quality_status')} actions={(plan.get('summary') or {}).get('action_count')}")
    return 0


def cmd_action_list(args: argparse.Namespace) -> int:
    print("action_id\tstatus\towner\tpriority\taction_type\tcount\ttitle")
    for action in list_iteration_actions(args.plan, status=args.status):
        print(
            "\t".join(
                _cell(value)
                for value in (
                    action.get("action_id"),
                    action.get("status"),
                    action.get("owner"),
                    action.get("priority"),
                    action.get("action_type"),
                    action.get("count"),
                    action.get("title"),
                )
            )
        )
    return 0


def cmd_action_update(args: argparse.Namespace) -> int:
    if args.status is None and args.owner is None and args.note is None:
        raise ValueError("at least one of --status, --owner, or --note is required")
    action = update_action(
        args.plan,
        action_id=args.action_id,
        status=args.status,
        owner=args.owner,
        note=args.note,
    )
    print(
        " ".join(
            [
                f"action_id={action.get('action_id')}",
                f"status={action.get('status')}",
                f"owner={_cell(action.get('owner'))}",
                f"updated_at={_cell(action.get('updated_at'))}",
            ]
        )
    )
    return 0


def cmd_launcher_plan(args: argparse.Namespace) -> int:
    plan = build_launch_plan(
        kind=args.kind,
        recipe_path=args.recipe,
        workspace=args.workspace,
        launcher=args.launcher,
        mode=args.mode,
        output_root=args.output_root,
        run_id=args.run_id,
        config_path=args.config,
        target=args.target,
        python=args.python,
        cwd=args.cwd,
        includes=args.include,
    )
    output = args.output or (args.output_root / "launch_plans" / f"{plan['plan_id']}.json")
    write_json(output, plan)
    script_path = output.with_suffix(".ps1")
    script_path.write_text(render_launch_script(plan), encoding="utf-8", newline="\n")
    print(f"launch_plan={output}")
    print(f"script={script_path}")
    run_commands = plan.get("commands", {}).get("run", [])
    if run_commands:
        print("next=" + run_commands[0])
    return 0


def cmd_job_submit(args: argparse.Namespace) -> int:
    status = submit_job(
        plan_path=args.plan,
        jobs_root=args.root,
        job_id=args.job_id,
        execute=args.execute,
        timeout_seconds=args.timeout_seconds,
    )
    print(f"job={status['job_dir']}")
    print(
        " ".join(
            [
                f"job_id={status.get('job_id')}",
                f"status={status.get('status')}",
                f"launcher={status.get('launcher')}",
                f"kind={status.get('kind')}",
                f"execute={status.get('execute')}",
            ]
        )
    )
    return 0 if status.get("status") in {"planned", "succeeded"} else 2


def cmd_job_status(args: argparse.Namespace) -> int:
    status = get_job_status(args.job)
    if args.json:
        print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(
            " ".join(
                [
                    f"job_id={status.get('job_id')}",
                    f"status={status.get('status')}",
                    f"launcher={status.get('launcher')}",
                    f"kind={status.get('kind')}",
                    f"commands_completed={status.get('commands_completed')}",
                    f"commands_failed={status.get('commands_failed')}",
                ]
            )
        )
        if status.get("collection_status"):
            print(f"collection_status={status.get('collection_status')}")
    return 0 if status.get("status") in {"planned", "running", "succeeded"} else 2


def cmd_job_run(args: argparse.Namespace) -> int:
    status = execute_job(args.job, timeout_seconds=args.timeout_seconds)
    print(
        " ".join(
            [
                f"job_id={status.get('job_id')}",
                f"status={status.get('status')}",
                f"commands_completed={status.get('commands_completed')}",
                f"commands_failed={status.get('commands_failed')}",
            ]
        )
    )
    return 0 if status.get("status") == "succeeded" else 2


def cmd_job_list(args: argparse.Namespace) -> int:
    print("job_id\tstatus\tlauncher\tkind\tmode\tcommands_completed\tcommands_failed\tupdated_at")
    for status in list_jobs(args.root):
        print(
            "\t".join(
                _cell(value)
                for value in (
                    status.get("job_id"),
                    status.get("status"),
                    status.get("launcher"),
                    status.get("kind"),
                    status.get("mode"),
                    status.get("commands_completed"),
                    status.get("commands_failed"),
                    status.get("updated_at"),
                )
            )
        )
    return 0


def cmd_job_logs(args: argparse.Namespace) -> int:
    for line in read_job_log(args.job, stream=args.stream, tail=args.tail):
        print(line)
    return 0


def cmd_job_collect(args: argparse.Namespace) -> int:
    status = collect_job(args.job, timeout_seconds=args.timeout_seconds)
    print(f"job_id={status.get('job_id')} collection_status={status.get('collection_status')}")
    return 0 if status.get("collection_status") in {"succeeded", "skipped"} else 2


def cmd_workbench_build(args: argparse.Namespace) -> int:
    result = build_workbench(
        root=args.root,
        output=args.output,
        registry_root=args.registry_root,
        rounds_root=args.rounds_root,
        train_runs_root=args.train_runs_root,
        eval_runs_root=args.eval_runs_root,
        jobs_root=args.jobs_root,
        title=args.title,
    )
    summary = result["summary"]
    print(f"workbench={result['output']}")
    print(f"data={result['data_output']}")
    print(
        " ".join(
            [
                f"rounds={summary.get('rounds')}",
                f"models={summary.get('models')}",
                f"actions={summary.get('actions')}",
                f"deployments={summary.get('deployments')}",
            ]
        )
    )
    return 0


def cmd_remote_add(args: argparse.Namespace) -> int:
    remote = add_remote(args.config, args.name, args.host, args.workdir, port=args.port, python=args.python)
    print(f"remote={args.name} host={remote['host']} workdir={remote['workdir']} config={args.config}")
    return 0


def cmd_remote_list(args: argparse.Namespace) -> int:
    remotes = list_remotes(args.config)
    if not remotes:
        print("no remotes configured")
        return 0
    for name, remote in sorted(remotes.items()):
        print(f"{name}\thost={remote['host']}\tport={remote.get('port', 22)}\tworkdir={remote['workdir']}")
    return 0


def cmd_remote_plan(args: argparse.Namespace) -> int:
    plan = build_remote_plan(args.config, args.target, args.run, args.recipe, workspace=args.workspace)
    output = args.output or (args.run / f"remote_plan_{args.target}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    script_path = output.with_suffix(".ps1")
    script_path.write_text(render_plan_script(plan), encoding="utf-8", newline="\n")
    print(f"remote_plan={output}")
    print(f"script={script_path}")
    print("next=" + plan["commands"]["prepare"][0])
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    store = LocalRunStore(args.run)
    tasks = _ingest_records(args.workspace, args.seed, store)
    traces = rollout_tasks(tasks, agent=args.agent)
    store.save_records("traces", "traces.jsonl", traces)
    store.save_report("rollout_summary", "rollout_summary.json", {"agent": args.agent, "traces": len(traces)})

    suite = read_jsonl(args.suite)
    eval_results, eval_report, accepted, rejected = evaluate_traces(traces, suite)
    store.save_records("eval_results", "eval_results.jsonl", eval_results)
    store.save_report("eval_report", "eval_report.json", eval_report)
    store.save_records("accepted", "accepted.jsonl", accepted)
    store.save_records("rejected", "rejected.jsonl", rejected)

    case_report = analyze_cases(eval_results, traces)
    store.save_report("case_report", "case_report.json", case_report)
    markdown_path = store.path("case_report.md")
    markdown_path.write_text(render_case_report_markdown(case_report), encoding="utf-8", newline="\n")
    store.update_artifact("case_report_md", "case_report.md")

    sft_path = store.path("sft.jsonl")
    preference_path = store.path("preference.jsonl")
    rl_path = store.path("rl_prompts.jsonl")
    write_jsonl(sft_path, export_sft(accepted))
    write_jsonl(preference_path, export_preference(case_report.get("cases", [])))
    write_jsonl(rl_path, export_rl_prompts(tasks))
    store.update_artifact("sft", "sft.jsonl")
    store.update_artifact("preference", "preference.jsonl")
    store.update_artifact("rl_prompts", "rl_prompts.jsonl")

    dry_run = build_train_dry_run(args.sft_recipe, workspace=args.workspace)
    dry_path = save_dry_run_manifest(dry_run, args.run.parent)
    store.update_artifact("sft_dry_run", _relative_to_run(dry_path, args.run))

    print(f"demo_run={args.run}")
    print(f"tasks={len(tasks)} traces={len(traces)} pass_rate={eval_report['pass_rate']}")
    print(f"failed_cases={case_report['total_failed_cases']} case_report={markdown_path}")
    print(f"sft={sft_path} preference={preference_path} rl_prompts={rl_path}")
    print(f"dry_run={dry_path}")
    return 0


def _require_workspace(store: LocalRunStore, workspace: str) -> None:
    manifest = store.load_manifest()
    if manifest.get("workspace_id") != workspace:
        raise ValueError(f"run workspace is {manifest.get('workspace_id')}, got {workspace}")


def _ingest_records(workspace: str, input_path: Path, store: LocalRunStore) -> list[dict[str, Any]]:
    records = read_jsonl(input_path)
    accepted: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for record in records:
        validation_errors = validate_task_sample(record)
        if record.get("workspace_id") != workspace:
            validation_errors.append("workspace_id does not match workspace")
        if validation_errors:
            errors.append({"record": record, "errors": validation_errors})
        else:
            accepted.append(record)
    store.init(workspace, str(input_path))
    store.save_records("tasks", "tasks.jsonl", accepted)
    store.save_records("ingest_errors", "ingest_errors.jsonl", errors)
    store.save_report(
        "ingest_summary",
        "ingest_summary.json",
        {"input": str(input_path), "accepted": len(accepted), "errors": len(errors)},
    )
    if errors:
        raise ValueError(f"demo input has invalid records: {len(errors)}")
    return accepted


def _count_jsonl(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _relative_to_run(path: Path, run_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(run_dir.resolve()))
    except ValueError:
        return str(path)


def _cell(value: Any) -> str:
    return "-" if value is None or value == "" else str(value)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
