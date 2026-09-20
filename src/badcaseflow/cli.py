from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .analyzer import analyze_cases, render_case_report_markdown
from .evaluation import evaluate_traces
from .exporters import export_preference, export_rl_prompts, export_sft
from .io_utils import read_json, read_jsonl, write_jsonl
from .remote import add_remote, build_remote_plan, list_remotes, render_plan_script
from .recipes import build_train_dry_run, save_dry_run_manifest
from .rollout import rollout_tasks
from .schemas import validate_agent_trace, validate_task_sample
from .store import LocalRunStore
from .workspace import init_workspace, load_workspace, remotes_path


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
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

    doctor = sub.add_parser("doctor", help="check local development environment")
    doctor.add_argument("--root", default=Path("."), type=Path)
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
    dry.set_defaults(func=cmd_train_dry_run)

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
    promote_dry.set_defaults(func=cmd_promote_dry_run)

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


def cmd_doctor(args: argparse.Namespace) -> int:
    print(f"python={sys.version.split()[0]}")
    print(f"root={args.root.resolve()}")
    try:
        workspace = load_workspace(args.root)
        print(f"workspace={workspace.get('workspace_id')}")
    except FileNotFoundError:
        print("workspace=not-initialized")
    for relative in (
        "examples/datasets/seed_tasks.jsonl",
        "examples/datasets/eval_tasks.jsonl",
        "examples/recipes/sft_llamafactory.example.yaml",
        "examples/recipes/opd_verl.example.yaml",
    ):
        print(f"{relative}={'ok' if (args.root / relative).exists() else 'missing'}")
    return 0


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
    manifest = build_train_dry_run(args.recipe, workspace=args.workspace)
    path = save_dry_run_manifest(manifest, args.output_root)
    print(f"dry_run={path}")
    print(f"preflight={manifest['preflight']['status']}")
    print("command=" + " ".join(str(part) for part in manifest["command"]))
    return 0 if manifest["preflight"]["status"] != "failed" else 2


def cmd_compare(args: argparse.Namespace) -> int:
    before = read_json(args.before)
    after = read_json(args.after)
    delta = round(float(after.get("pass_rate", 0.0)) - float(before.get("pass_rate", 0.0)), 4)
    print(f"before_pass_rate={before.get('pass_rate')} after_pass_rate={after.get('pass_rate')} delta={delta}")
    return 0


def cmd_promote_dry_run(args: argparse.Namespace) -> int:
    report = read_json(args.eval_report)
    pass_rate = float(report.get("pass_rate", 0.0))
    decision = "approved" if pass_rate >= args.min_pass_rate and int(report.get("failed", 0)) == 0 else "blocked"
    print(f"model={args.model} decision={decision} pass_rate={pass_rate} min_pass_rate={args.min_pass_rate}")
    return 0 if decision == "approved" else 2


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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
