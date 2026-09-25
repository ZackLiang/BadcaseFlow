from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import (
    StageSpec,
    TrainingAdapter,
    collect_data_path_warnings,
    collect_stage_errors,
    collect_stage_warnings,
    deep_get,
    find_first,
    first_model_path,
    make_preflight,
    output_dir,
)


class LlamaFactoryAdapter(TrainingAdapter):
    backend = "llamafactory"
    display_name = "LLaMA-Factory"
    stages = (
        StageSpec("pt", "预训练", "继续预训练或领域语料适配", aliases=("pretrain",)),
        StageSpec("sft", "监督微调", "指令、工具调用轨迹和多轮对话 SFT"),
        StageSpec("rm", "奖励模型", "偏好数据训练 reward model", aliases=("reward", "reward_model")),
        StageSpec("ppo", "PPO", "基于 reward model 或规则奖励的 PPO"),
        StageSpec("dpo", "偏好优化", "DPO 以及 pref_loss=orpo/simpo 等偏好优化变体", aliases=("orpo", "simpo")),
        StageSpec("kto", "KTO", "用正负反馈样本做 Kahneman-Tversky Optimization"),
        StageSpec("export", "模型导出", "合并 LoRA 或导出可部署权重"),
    )
    stage_aliases = {
        "pretrain": "pt",
        "reward": "rm",
        "reward_model": "rm",
        "orpo": "dpo",
        "simpo": "dpo",
        "preference": "dpo",
    }

    def build_command(self, recipe: dict[str, Any], recipe_path: Path, stage: str) -> list[str]:
        # LLaMA-Factory 直接读取原生 YAML，因此 recipe 文件必须保留其字段命名。
        cli = find_first(recipe, ("launcher", "command"), ("launcher", "cli")) or "llamafactory-cli"
        subcommand = "export" if stage == "export" else "train"
        return [str(cli), subcommand, str(recipe_path)]

    def preflight(self, recipe: dict[str, Any], recipe_path: Path, stage: str) -> dict[str, Any]:
        errors = collect_stage_errors(self, stage)
        warnings = collect_stage_warnings(self, recipe, stage)
        warnings.extend(collect_data_path_warnings(recipe, recipe_path))

        if first_model_path(recipe) is None and deep_get(recipe, ("adapter_name_or_path",)) is None:
            warnings.append("model path is not declared; set model.base_model or model_name_or_path")

        if stage != "export" and not _has_training_dataset(recipe):
            warnings.append("training dataset is not declared; set data.train_file or dataset")

        dataset_dir = recipe.get("dataset_dir")
        if dataset_dir not in (None, "") and not _path_exists(str(dataset_dir), recipe_path):
            warnings.append(f"dataset_dir does not exist yet: {dataset_dir}")

        if output_dir(recipe) is None:
            warnings.append("output directory is not declared; set outputs.output_dir or output_dir")

        if stage in {"orpo", "simpo"}:
            warnings.append("LLaMA-Factory usually uses stage=dpo plus pref_loss=orpo/simpo")

        return make_preflight(errors, warnings)


def _has_training_dataset(recipe: dict[str, Any]) -> bool:
    return any(
        value not in (None, "")
        for value in (
            deep_get(recipe, ("data", "train_file")),
            deep_get(recipe, ("data", "train_files")),
            deep_get(recipe, ("data", "dataset")),
            deep_get(recipe, ("dataset",)),
            deep_get(recipe, ("train_file",)),
        )
    )


def _path_exists(value: str, recipe_path: Path) -> bool:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.exists()
    roots = (Path.cwd(), recipe_path.parent, recipe_path.parent / ".." / "..")
    return any((root / candidate).resolve().exists() for root in roots)
