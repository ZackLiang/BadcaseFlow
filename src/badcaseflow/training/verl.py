from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import (
    StageSpec,
    TrainingAdapter,
    add_override,
    append_user_overrides,
    collect_data_path_warnings,
    collect_stage_errors,
    collect_stage_warnings,
    deep_get,
    find_first,
    first_model_path,
    make_preflight,
    output_dir,
)


class VerlAdapter(TrainingAdapter):
    backend = "verl"
    display_name = "verl"
    stages = (
        StageSpec("sft", "监督微调", "verl.trainer.sft_trainer，适合 parquet 多轮 SFT 数据"),
        StageSpec("ppo", "PPO", "verl.trainer.main_ppo 的 PPO/RLHF 训练入口"),
        StageSpec("grpo", "GRPO", "无 critic 的 group relative policy optimization"),
        StageSpec("dapo", "DAPO", "基于 verl PPO 主入口的 DAPO 变体"),
        StageSpec("rloo", "RLOO", "Leave-one-out policy optimization 变体"),
        StageSpec("remax", "ReMax", "轻量 RL 优化变体"),
        StageSpec("reinforce_plus_plus", "Reinforce++", "verl 示例中的 Reinforce++ 变体", aliases=("reinforce++",)),
        StageSpec("gspo", "GSPO", "verl 示例中的 GSPO 变体"),
        StageSpec("opd", "OPD", "在 main_ppo 上开启 distillation 配置的在线策略蒸馏", aliases=("distill", "distillation")),
        StageSpec("eval", "离线评测", "verl.trainer.main_eval 离线评测入口", aliases=("evaluate",)),
    )
    stage_aliases = {
        "distill": "opd",
        "distillation": "opd",
        "online_policy_distillation": "opd",
        "reinforce++": "reinforce_plus_plus",
        "reinforce-plus-plus": "reinforce_plus_plus",
        "evaluate": "eval",
        "rl": "ppo",
    }

    def build_command(self, recipe: dict[str, Any], recipe_path: Path, stage: str) -> list[str]:
        python = find_first(recipe, ("launcher", "python"), ("python",)) or "python"
        module = _module_for_stage(stage)
        command = [str(python), "-m", module]
        if stage == "sft":
            _add_sft_overrides(command, recipe)
        elif stage == "eval":
            _add_eval_overrides(command, recipe)
        else:
            _add_ppo_family_overrides(command, recipe, stage)
        append_user_overrides(command, recipe)
        return command

    def preflight(self, recipe: dict[str, Any], recipe_path: Path, stage: str) -> dict[str, Any]:
        errors = collect_stage_errors(self, stage)
        warnings = collect_stage_warnings(self, recipe, stage)
        warnings.extend(collect_data_path_warnings(recipe, recipe_path))

        if stage == "eval":
            if find_first(recipe, ("data", "path"), ("data", "eval_file"), ("data", "val_file")) is None:
                warnings.append("eval data is not declared; set data.path or data.eval_file")
            return make_preflight(errors, warnings)

        if first_model_path(recipe) is None:
            warnings.append("student/model path is not declared; set student.model or model.base_model")
        if find_first(recipe, ("data", "train_file"), ("data", "train_files")) is None:
            warnings.append("training data is not declared; set data.train_file or data.train_files")
        if output_dir(recipe) is None:
            warnings.append("output directory is not declared; set outputs.output_dir")
        if stage == "opd" and find_first(recipe, ("teacher", "model"), ("distillation", "teacher_model")) is None:
            warnings.append("OPD teacher model is not declared; set teacher.model")

        return make_preflight(errors, warnings)


def _module_for_stage(stage: str) -> str:
    if stage == "sft":
        return "verl.trainer.sft_trainer"
    if stage == "eval":
        return "verl.trainer.main_eval"
    return "verl.trainer.main_ppo"


def _add_sft_overrides(command: list[str], recipe: dict[str, Any]) -> None:
    data = recipe.get("data", {}) if isinstance(recipe.get("data"), dict) else {}
    training = recipe.get("training", {}) if isinstance(recipe.get("training"), dict) else {}

    add_override(command, "model.path", first_model_path(recipe))
    add_override(command, "data.train_files", data.get("train_files") or data.get("train_file"), list_value=True)
    add_override(command, "data.val_files", data.get("val_files") or data.get("val_file"), list_value=True)
    add_override(command, "data.train_batch_size", training.get("train_batch_size"))
    add_override(command, "data.micro_batch_size_per_gpu", training.get("micro_batch_size_per_gpu"))
    add_override(command, "data.max_length", data.get("max_length"))
    add_override(command, "trainer.total_epochs", training.get("total_epochs") or training.get("num_train_epochs"))
    add_override(command, "trainer.logger", training.get("logger"))
    add_override(command, "trainer.project_name", training.get("project_name"))
    add_override(command, "trainer.experiment_name", training.get("experiment_name"))
    add_override(command, "trainer.save_freq", training.get("save_freq") or training.get("save_steps"))
    add_override(command, "trainer.test_freq", training.get("test_freq") or training.get("eval_steps"))
    add_override(command, "trainer.default_local_dir", output_dir(recipe))


def _add_ppo_family_overrides(command: list[str], recipe: dict[str, Any], stage: str) -> None:
    data = recipe.get("data", {}) if isinstance(recipe.get("data"), dict) else {}
    training = recipe.get("training", {}) if isinstance(recipe.get("training"), dict) else {}
    rollout = recipe.get("rollout", {}) if isinstance(recipe.get("rollout"), dict) else {}
    algorithm = recipe.get("algorithm", {}) if isinstance(recipe.get("algorithm"), dict) else {}

    estimator = algorithm.get("adv_estimator")
    if estimator is None and stage not in {"ppo", "opd"}:
        estimator = stage
    if estimator is None and stage == "opd":
        estimator = "grpo"
    add_override(command, "algorithm.adv_estimator", estimator)

    add_override(command, "data.train_files", data.get("train_files") or data.get("train_file"), list_value=True)
    add_override(command, "data.val_files", data.get("val_files") or data.get("val_file"), list_value=True)
    add_override(command, "data.train_batch_size", training.get("train_batch_size"))
    add_override(command, "data.max_prompt_length", data.get("max_prompt_length"))
    add_override(command, "data.max_response_length", data.get("max_response_length"))

    add_override(command, "actor_rollout_ref.model.path", first_model_path(recipe))
    add_override(command, "actor_rollout_ref.actor.optim.lr", training.get("learning_rate") or training.get("actor_lr"))
    add_override(command, "actor_rollout_ref.actor.ppo_mini_batch_size", training.get("ppo_mini_batch_size"))
    add_override(command, "actor_rollout_ref.rollout.name", rollout.get("engine") or rollout.get("name"))
    add_override(command, "actor_rollout_ref.rollout.tensor_model_parallel_size", rollout.get("tensor_parallel_size"))
    add_override(command, "actor_rollout_ref.rollout.gpu_memory_utilization", rollout.get("gpu_memory_utilization"))
    add_override(command, "actor_rollout_ref.rollout.n", rollout.get("n"))

    add_override(command, "trainer.logger", training.get("logger"))
    add_override(command, "trainer.project_name", training.get("project_name"))
    add_override(command, "trainer.experiment_name", training.get("experiment_name"))
    add_override(command, "trainer.total_epochs", training.get("total_epochs") or training.get("num_train_epochs"))
    add_override(command, "trainer.n_gpus_per_node", training.get("n_gpus_per_node") or training.get("gpus_per_node"))
    add_override(command, "trainer.nnodes", training.get("nnodes"))
    add_override(command, "trainer.save_freq", training.get("save_freq") or training.get("save_steps"))
    add_override(command, "trainer.test_freq", training.get("test_freq") or training.get("eval_steps"))
    add_override(command, "trainer.default_local_dir", output_dir(recipe))

    if stage == "opd":
        _add_distillation_overrides(command, recipe)


def _add_distillation_overrides(command: list[str], recipe: dict[str, Any]) -> None:
    teacher = recipe.get("teacher", {}) if isinstance(recipe.get("teacher"), dict) else {}
    distillation = recipe.get("distillation", {}) if isinstance(recipe.get("distillation"), dict) else {}
    rollout = recipe.get("rollout", {}) if isinstance(recipe.get("rollout"), dict) else {}

    add_override(command, "distillation.enabled", True)
    add_override(command, "distillation.n_gpus_per_node", teacher.get("gpus") or distillation.get("n_gpus_per_node"))
    add_override(command, "distillation.nnodes", teacher.get("nnodes") or distillation.get("nnodes"))
    add_override(command, "distillation.teacher_models.teacher_model.model_path", teacher.get("model"))
    add_override(
        command,
        "distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size",
        teacher.get("tensor_parallel_size") or rollout.get("teacher_tensor_parallel_size"),
    )
    add_override(
        command,
        "distillation.teacher_models.teacher_model.inference.name",
        teacher.get("engine") or rollout.get("engine") or rollout.get("name"),
    )
    add_override(
        command,
        "distillation.teacher_models.teacher_model.inference.gpu_memory_utilization",
        teacher.get("gpu_memory_utilization") or rollout.get("teacher_gpu_memory_utilization"),
    )
    add_override(command, "distillation.distillation_loss.loss_mode", distillation.get("loss_mode"))
    add_override(command, "distillation.distillation_loss.topk", distillation.get("topk"))
    add_override(command, "distillation.distillation_loss.use_policy_gradient", distillation.get("use_policy_gradient"))
    add_override(command, "distillation.distillation_loss.use_task_rewards", distillation.get("use_task_rewards"))


def _add_eval_overrides(command: list[str], recipe: dict[str, Any]) -> None:
    data = recipe.get("data", {}) if isinstance(recipe.get("data"), dict) else {}
    add_override(command, "data.path", data.get("path") or data.get("eval_file") or data.get("val_file"))
    add_override(command, "data.response_key", data.get("response_key"))
    add_override(command, "data.data_source_key", data.get("data_source_key"))
    add_override(command, "data.reward_model_key", data.get("reward_model_key"))
