from __future__ import annotations

import json
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .datasets import build_dataset_version
from .io_utils import read_json, write_json
from .workspace import DEFAULT_CONFIG_DIR


# registry 保存数据、训练产物、模型、评测、门禁和部署之间的关系。
# 它不是 checkpoint 存储本身，真实产物仍保留在用户指定的目录。
REGISTRY_FILENAME = "registry.json"


def registry_path(root: Path) -> Path:
    return root / DEFAULT_CONFIG_DIR / REGISTRY_FILENAME


def load_registry(root: Path) -> dict[str, Any]:
    path = registry_path(root)
    if not path.exists():
        return _empty_registry()
    registry = read_json(path)
    registry.setdefault("version", 1)
    registry.setdefault("artifacts", [])
    registry.setdefault("models", [])
    registry.setdefault("evaluations", [])
    registry.setdefault("promotions", [])
    registry.setdefault("datasets", [])
    registry.setdefault("deployments", [])
    return registry


def save_registry(root: Path, registry: dict[str, Any]) -> None:
    registry["updated_at"] = _now()
    write_json(registry_path(root), registry)


def index_train_run(root: Path, run_dir: Path) -> dict[str, Any]:
    status = _load_train_status(run_dir)
    artifact_manifest = _load_artifact_manifest(run_dir)
    registry = load_registry(root)
    existing = {
        item["artifact_id"]: item
        for item in registry.get("artifacts", [])
        if isinstance(item, dict) and "artifact_id" in item
    }
    for artifact in artifact_manifest.get("artifacts", []):
        artifact_id = f"{status.get('train_run_id')}:{artifact.get('name')}"
        path = run_dir / str(artifact.get("path"))
        existing[artifact_id] = {
            "artifact_id": artifact_id,
            "name": artifact.get("name"),
            "kind": artifact.get("kind"),
            "source_type": "train_run",
            "train_run_id": status.get("train_run_id"),
            "workspace_id": status.get("workspace_id"),
            "backend": status.get("backend"),
            "stage": status.get("stage"),
            "recipe_id": status.get("recipe_id"),
            "status": status.get("status"),
            "diagnosis": (status.get("diagnosis") or {}).get("category"),
            "path": str(path),
            "relative_path": str(artifact.get("path")),
            "bytes": artifact.get("bytes", 0),
            "exists": bool(artifact.get("exists")),
            "updated_at": _now(),
        }
    registry["artifacts"] = sorted(existing.values(), key=lambda item: item["artifact_id"])
    save_registry(root, registry)
    return {
        "indexed": len(artifact_manifest.get("artifacts", [])),
        "train_run_id": status.get("train_run_id"),
        "registry": str(registry_path(root)),
    }


def index_eval_run(root: Path, run_dir: Path) -> dict[str, Any]:
    status = _load_eval_status(run_dir)
    artifact_manifest = _load_eval_artifact_manifest(run_dir)
    registry = load_registry(root)
    existing = {
        item["artifact_id"]: item
        for item in registry.get("artifacts", [])
        if isinstance(item, dict) and "artifact_id" in item
    }
    for artifact in artifact_manifest.get("artifacts", []):
        artifact_id = f"{status.get('eval_run_id')}:{artifact.get('name')}"
        path = run_dir / str(artifact.get("path"))
        existing[artifact_id] = {
            "artifact_id": artifact_id,
            "name": artifact.get("name"),
            "kind": artifact.get("kind"),
            "source_type": "eval_run",
            "eval_run_id": status.get("eval_run_id"),
            "workspace_id": status.get("workspace_id"),
            "backend": status.get("backend"),
            "stage": status.get("stage"),
            "recipe_id": status.get("recipe_id"),
            "status": status.get("status"),
            "quality_status": status.get("quality_status"),
            "path": str(path),
            "relative_path": str(artifact.get("path")),
            "bytes": artifact.get("bytes", 0),
            "exists": bool(artifact.get("exists")),
            "updated_at": _now(),
        }
    registry["artifacts"] = sorted(existing.values(), key=lambda item: item["artifact_id"])
    save_registry(root, registry)
    return {
        "indexed": len(artifact_manifest.get("artifacts", [])),
        "eval_run_id": status.get("eval_run_id"),
        "registry": str(registry_path(root)),
    }


def register_model(
    root: Path,
    *,
    model_id: str,
    run_dir: Path,
    model_path: str | None = None,
    description: str | None = None,
    allow_non_succeeded: bool = False,
    dataset_version_ids: list[str] | None = None,
) -> dict[str, Any]:
    if not model_id.strip():
        raise ValueError("model_id is required")
    status = _load_train_status(run_dir)
    if status.get("status") != "succeeded" and not allow_non_succeeded:
        raise ValueError(f"train run is not succeeded: {status.get('status')}")

    registry = load_registry(root)
    models = {
        item["model_id"]: item
        for item in registry.get("models", [])
        if isinstance(item, dict) and "model_id" in item
    }
    source_manifest = status.get("source_manifest") if isinstance(status.get("source_manifest"), dict) else {}
    artifact_info = source_manifest.get("artifacts") if isinstance(source_manifest.get("artifacts"), dict) else {}
    resolved_model_path = model_path or artifact_info.get("output_dir") or str(run_dir)
    latest_metrics = _load_latest_metrics(run_dir)
    linked_dataset_ids = list(models.get(model_id, {}).get("dataset_version_ids", []))
    for dataset_version_id in dataset_version_ids or []:
        if dataset_version_id not in linked_dataset_ids:
            linked_dataset_ids.append(dataset_version_id)
    model = {
        "model_id": model_id,
        "description": description,
        "status": "candidate",
        "train_run_id": status.get("train_run_id"),
        "workspace_id": status.get("workspace_id"),
        "backend": status.get("backend"),
        "stage": status.get("stage"),
        "recipe_id": status.get("recipe_id"),
        "model_path": resolved_model_path,
        "dataset_version_ids": linked_dataset_ids,
        "diagnosis": (status.get("diagnosis") or {}).get("category"),
        "metrics": latest_metrics,
        "registered_at": models.get(model_id, {}).get("registered_at") or _now(),
        "updated_at": _now(),
    }
    models[model_id] = model
    registry["models"] = sorted(models.values(), key=lambda item: item["model_id"])
    save_registry(root, registry)
    return model


def attach_eval_report(
    root: Path,
    *,
    model_id: str,
    eval_report_path: Path,
    min_pass_rate: float = 0.8,
    allow_failed_cases: bool = False,
    eval_run_dir: Path | None = None,
) -> dict[str, Any]:
    if not model_id.strip():
        raise ValueError("model_id is required")
    # 先计算门禁结果，再把同一 decision 同步到 evaluation、promotion 和 model。
    report = read_json(eval_report_path)
    decision = decide_promotion(report, min_pass_rate=min_pass_rate, allow_failed_cases=allow_failed_cases)

    registry = load_registry(root)
    models = {
        item["model_id"]: item
        for item in registry.get("models", [])
        if isinstance(item, dict) and "model_id" in item
    }
    if model_id not in models:
        raise KeyError(f"model not found: {model_id}")

    summary = _eval_summary(report)
    eval_status = _load_eval_status(eval_run_dir) if eval_run_dir is not None else {}
    fingerprint = _short_hash(
        {
            "model_id": model_id,
            "eval_report_path": str(eval_report_path),
            "eval_run_id": eval_status.get("eval_run_id"),
            "summary": summary,
            "min_pass_rate": min_pass_rate,
            "allow_failed_cases": allow_failed_cases,
        }
    )
    evaluation_id = f"eval-{_slug(model_id)}-{fingerprint}"
    promotion_id = f"promotion-{_slug(model_id)}-{fingerprint}"
    now = _now()

    evaluations = {
        item["evaluation_id"]: item
        for item in registry.get("evaluations", [])
        if isinstance(item, dict) and "evaluation_id" in item
    }
    previous_evaluation = evaluations.get(evaluation_id, {})
    evaluation = {
        "evaluation_id": evaluation_id,
        "model_id": model_id,
        "train_run_id": models[model_id].get("train_run_id"),
        "eval_run_id": eval_status.get("eval_run_id"),
        "workspace_id": models[model_id].get("workspace_id"),
        "backend": eval_status.get("backend"),
        "stage": eval_status.get("stage"),
        "recipe_id": eval_status.get("recipe_id"),
        "report_path": str(eval_report_path),
        "report_name": eval_report_path.name,
        "summary": summary,
        "failure_counts": report.get("failure_counts", {}),
        "decision": decision["decision"],
        "gate": {
            "min_pass_rate": min_pass_rate,
            "allow_failed_cases": allow_failed_cases,
            "reasons": decision["reasons"],
        },
        "attached_at": previous_evaluation.get("attached_at") or now,
        "updated_at": now,
    }
    evaluations[evaluation_id] = evaluation

    promotions = {
        item["promotion_id"]: item
        for item in registry.get("promotions", [])
        if isinstance(item, dict) and "promotion_id" in item
    }
    previous_promotion = promotions.get(promotion_id, {})
    promotion = {
        "promotion_id": promotion_id,
        "model_id": model_id,
        "evaluation_id": evaluation_id,
        "decision": decision["decision"],
        "pass_rate": decision["pass_rate"],
        "failed": decision["failed"],
        "min_pass_rate": min_pass_rate,
        "allow_failed_cases": allow_failed_cases,
        "reasons": decision["reasons"],
        "created_at": previous_promotion.get("created_at") or now,
        "updated_at": now,
    }
    promotions[promotion_id] = promotion

    model = dict(models[model_id])
    evaluation_ids = list(model.get("evaluation_ids", []))
    if evaluation_id not in evaluation_ids:
        evaluation_ids.append(evaluation_id)
    model.update(
        {
            "status": decision["decision"],
            "evaluation_ids": evaluation_ids,
            "latest_evaluation_id": evaluation_id,
            "latest_promotion_id": promotion_id,
            "latest_eval_report": summary,
            "promotion_reasons": decision["reasons"],
            "updated_at": now,
        }
    )
    models[model_id] = model

    registry["models"] = sorted(models.values(), key=lambda item: item["model_id"])
    registry["evaluations"] = sorted(evaluations.values(), key=lambda item: item["evaluation_id"])
    registry["promotions"] = sorted(promotions.values(), key=lambda item: item["promotion_id"])
    save_registry(root, registry)
    return {"model": model, "evaluation": evaluation, "promotion": promotion, "decision": decision}


def attach_eval_run(
    root: Path,
    *,
    model_id: str,
    eval_run_dir: Path,
    min_pass_rate: float = 0.8,
    allow_failed_cases: bool = False,
) -> dict[str, Any]:
    index_eval_run(root, eval_run_dir)
    report_path = eval_run_dir / "eval_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"eval report not found: {report_path}")
    return attach_eval_report(
        root,
        model_id=model_id,
        eval_report_path=report_path,
        min_pass_rate=min_pass_rate,
        allow_failed_cases=allow_failed_cases,
        eval_run_dir=eval_run_dir,
    )


def decide_promotion(
    report: dict[str, Any],
    *,
    min_pass_rate: float = 0.8,
    allow_failed_cases: bool = False,
) -> dict[str, Any]:
    summary = _eval_summary(report)
    pass_rate = summary["pass_rate"]
    failed = summary["failed"]
    reasons: list[str] = []
    if pass_rate < min_pass_rate:
        reasons.append("pass_rate_below_threshold")
    if failed > 0 and not allow_failed_cases:
        reasons.append("failed_cases_present")
    decision = "approved" if not reasons else "blocked"
    return {
        "decision": decision,
        "pass_rate": pass_rate,
        "failed": failed,
        "min_pass_rate": min_pass_rate,
        "allow_failed_cases": allow_failed_cases,
        "reasons": reasons,
    }


def list_artifacts(root: Path) -> list[dict[str, Any]]:
    return list(load_registry(root).get("artifacts", []))


def list_models(root: Path) -> list[dict[str, Any]]:
    return list(load_registry(root).get("models", []))


def list_evaluations(root: Path) -> list[dict[str, Any]]:
    return list(load_registry(root).get("evaluations", []))


def list_promotions(root: Path) -> list[dict[str, Any]]:
    return list(load_registry(root).get("promotions", []))


def register_deployment(root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    deployment_id = str(plan.get("deployment_id") or "").strip()
    model_id = str(plan.get("model_id") or "").strip()
    if not deployment_id:
        raise ValueError("deployment_id is required")
    if not model_id:
        raise ValueError("model_id is required")

    registry = load_registry(root)
    models = {
        item["model_id"]: item
        for item in registry.get("models", [])
        if isinstance(item, dict) and "model_id" in item
    }
    if model_id not in models:
        raise KeyError(f"model not found: {model_id}")

    deployments = {
        item["deployment_id"]: item
        for item in registry.get("deployments", [])
        if isinstance(item, dict) and "deployment_id" in item
    }
    previous = deployments.get(deployment_id, {})
    record = {
        **plan,
        "registered_at": previous.get("registered_at") or _now(),
        "updated_at": _now(),
    }
    deployments[deployment_id] = record

    model = dict(models[model_id])
    deployment_ids = list(model.get("deployment_ids", []))
    if deployment_id not in deployment_ids:
        deployment_ids.append(deployment_id)
    model.update(
        {
            "deployment_ids": deployment_ids,
            "latest_deployment_id": deployment_id,
            "updated_at": _now(),
        }
    )
    models[model_id] = model

    registry["deployments"] = sorted(deployments.values(), key=lambda item: item["deployment_id"])
    registry["models"] = sorted(models.values(), key=lambda item: item["model_id"])
    save_registry(root, registry)
    manifest_path = root / DEFAULT_CONFIG_DIR / "deployments" / f"{deployment_id}.json"
    write_json(manifest_path, record)
    return record


def list_deployments(root: Path) -> list[dict[str, Any]]:
    return list(load_registry(root).get("deployments", []))


def get_deployment(root: Path, deployment_id: str) -> dict[str, Any]:
    for deployment in list_deployments(root):
        if deployment.get("deployment_id") == deployment_id:
            return deployment
    raise KeyError(f"deployment not found: {deployment_id}")


def register_dataset(
    root: Path,
    *,
    path: Path,
    kind: str,
    dataset_id: str | None = None,
    workspace_id: str | None = None,
    description: str | None = None,
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = build_dataset_version(
        path,
        kind=kind,
        dataset_id=dataset_id,
        workspace_id=workspace_id,
        description=description,
        source=source,
    )
    return register_dataset_manifest(root, manifest)


def register_dataset_manifest(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    dataset_version_id = str(manifest.get("dataset_version_id") or "").strip()
    if not dataset_version_id:
        raise ValueError("dataset_version_id is required")
    registry = load_registry(root)
    datasets = {
        item["dataset_version_id"]: item
        for item in registry.get("datasets", [])
        if isinstance(item, dict) and "dataset_version_id" in item
    }
    previous = datasets.get(dataset_version_id, {})
    record = {
        **manifest,
        "registered_at": previous.get("registered_at") or _now(),
        "updated_at": _now(),
    }
    datasets[dataset_version_id] = record
    registry["datasets"] = sorted(datasets.values(), key=lambda item: item["dataset_version_id"])
    save_registry(root, registry)
    manifest_path = root / DEFAULT_CONFIG_DIR / "datasets" / f"{dataset_version_id}.json"
    write_json(manifest_path, record)
    return record


def list_datasets(root: Path) -> list[dict[str, Any]]:
    return list(load_registry(root).get("datasets", []))


def get_dataset(root: Path, dataset_version_id: str) -> dict[str, Any]:
    for dataset in list_datasets(root):
        if dataset.get("dataset_version_id") == dataset_version_id or dataset.get("dataset_id") == dataset_version_id:
            return dataset
    raise KeyError(f"dataset not found: {dataset_version_id}")


def get_model(root: Path, model_id: str) -> dict[str, Any]:
    for model in list_models(root):
        if model.get("model_id") == model_id:
            return model
    raise KeyError(f"model not found: {model_id}")


def get_model_lineage(root: Path, model_id: str) -> dict[str, Any]:
    model = get_model(root, model_id)
    train_run_id = model.get("train_run_id")
    dataset_ids = set(model.get("dataset_version_ids") or [])
    return {
        "model": model,
        "datasets": [
            dataset for dataset in list_datasets(root) if dataset.get("dataset_version_id") in dataset_ids
        ],
        "train_artifacts": [
            artifact for artifact in list_artifacts(root) if artifact.get("train_run_id") == train_run_id
        ],
        "evaluations": [
            evaluation for evaluation in list_evaluations(root) if evaluation.get("model_id") == model_id
        ],
        "promotions": [
            promotion for promotion in list_promotions(root) if promotion.get("model_id") == model_id
        ],
        "deployments": [
            deployment for deployment in list_deployments(root) if deployment.get("model_id") == model_id
        ],
    }


def _load_train_status(run_dir: Path) -> dict[str, Any]:
    status_path = run_dir / "status.json"
    manifest_path = run_dir / "manifest.json"
    if status_path.exists():
        return read_json(status_path)
    if manifest_path.exists():
        return read_json(manifest_path)
    raise FileNotFoundError(f"train run status not found: {run_dir}")


def _load_eval_status(run_dir: Path | None) -> dict[str, Any]:
    if run_dir is None:
        return {}
    status_path = run_dir / "status.json"
    manifest_path = run_dir / "manifest.json"
    if status_path.exists():
        return read_json(status_path)
    if manifest_path.exists():
        return read_json(manifest_path)
    raise FileNotFoundError(f"eval run status not found: {run_dir}")


def _load_artifact_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "artifacts.json"
    if not path.exists():
        status = _load_train_status(run_dir)
        artifacts = []
        for name, relative in sorted((status.get("artifacts") or {}).items()):
            artifact_path = run_dir / str(relative)
            artifacts.append(
                {
                    "name": name,
                    "path": relative,
                    "exists": artifact_path.exists(),
                    "bytes": artifact_path.stat().st_size if artifact_path.exists() else 0,
                    "kind": "metadata" if str(relative).endswith(".json") else "file",
                }
            )
        return {"train_run_id": status.get("train_run_id"), "status": status.get("status"), "artifacts": artifacts}
    return read_json(path)


def _load_eval_artifact_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "artifacts.json"
    if not path.exists():
        status = _load_eval_status(run_dir)
        artifacts = []
        for name, relative in sorted((status.get("artifacts") or {}).items()):
            artifact_path = run_dir / str(relative)
            artifacts.append(
                {
                    "name": name,
                    "path": relative,
                    "exists": artifact_path.exists(),
                    "bytes": artifact_path.stat().st_size if artifact_path.exists() else 0,
                    "kind": "metadata" if str(relative).endswith(".json") else "records"
                    if str(relative).endswith(".jsonl")
                    else "log"
                    if str(relative).endswith(".log")
                    else "file",
                }
            )
        return {"eval_run_id": status.get("eval_run_id"), "status": status.get("status"), "artifacts": artifacts}
    return read_json(path)


def _load_latest_metrics(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return {}
    latest: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        values = record.get("values")
        if isinstance(values, dict):
            latest.update(values)
    return latest


def _empty_registry() -> dict[str, Any]:
    now = _now()
    return {
        "version": 1,
        "created_at": now,
        "updated_at": now,
        "artifacts": [],
        "models": [],
        "evaluations": [],
        "promotions": [],
        "datasets": [],
        "deployments": [],
    }


def _eval_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "total": _as_int(report.get("total")),
        "passed": _as_int(report.get("passed")),
        "failed": _as_int(report.get("failed")),
        "pass_rate": round(_as_float(report.get("pass_rate")), 4),
    }


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _short_hash(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()).strip("-").lower()
    return slug or "model"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
