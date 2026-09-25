from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .io_utils import read_jsonl, write_json, write_jsonl


# GSM8K 转换器只负责协议转换和可选 parquet 写出，不负责下载数据集。
# 这样本地测试可以完全离线，远程环境也可以把已有 JSONL 作为输入。
DEFAULT_WORKSPACE_ID = "gsm8k-agent"
DEFAULT_SOURCE_NAME = "gsm8k"
DEFAULT_SUITE_ID = "gsm8k-math-v0.1"
DEFAULT_SYSTEM_PROMPT = "You are a careful math agent. Solve step by step and end with '#### <answer>'."
DEFAULT_PROMPT_SUFFIX = "Let's think step by step. Put the final answer after ####."


@dataclass(frozen=True)
class Gsm8kExample:
    question: str
    solution: str
    answer: str
    raw_id: str | None = None


def prepare_gsm8k(
    input_path: Path,
    output_dir: Path,
    *,
    val_input_path: Path | None = None,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    source_name: str = DEFAULT_SOURCE_NAME,
    suite_id: str = DEFAULT_SUITE_ID,
    train_ratio: float = 0.8,
    limit: int | None = None,
    val_limit: int | None = None,
    write_parquet: bool = False,
) -> dict[str, Any]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be greater than 0")
    if val_limit is not None and val_limit <= 0:
        raise ValueError("val_limit must be greater than 0")

    # 先统一成标准样本对象，再一次性投影到平台、SFT、RL 和 verl 四种格式。
    train_examples, val_examples = _load_splits(
        input_path,
        val_input_path=val_input_path,
        train_ratio=train_ratio,
        limit=limit,
        val_limit=val_limit,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    task_records = _task_records(
        train_examples,
        split="train",
        workspace_id=workspace_id,
        source_name=source_name,
    ) + _task_records(
        val_examples,
        split="val",
        workspace_id=workspace_id,
        source_name=source_name,
    )
    eval_records = _eval_records(val_examples, suite_id=suite_id)
    sft_train = _sft_records(train_examples, split="train", source_name=source_name)
    sft_val = _sft_records(val_examples, split="val", source_name=source_name)
    rl_train = _rl_records(train_examples, split="train", source_name=source_name)
    rl_val = _rl_records(val_examples, split="val", source_name=source_name)
    verl_train = _verl_records(train_examples, split="train", source_name=source_name)
    verl_val = _verl_records(val_examples, split="val", source_name=source_name)

    paths = {
        "task_samples": output_dir / "task_samples.jsonl",
        "eval_suite": output_dir / "eval_suite.jsonl",
        "sft_train": output_dir / "sft_train.jsonl",
        "sft_val": output_dir / "sft_val.jsonl",
        "rl_train": output_dir / "rl_train.jsonl",
        "rl_val": output_dir / "rl_val.jsonl",
        "verl_train_jsonl": output_dir / "verl_train.jsonl",
        "verl_val_jsonl": output_dir / "verl_val.jsonl",
        "dataset_info": output_dir / "dataset_info.json",
        "manifest": output_dir / "manifest.json",
    }
    write_jsonl(paths["task_samples"], task_records)
    write_jsonl(paths["eval_suite"], eval_records)
    write_jsonl(paths["sft_train"], sft_train)
    write_jsonl(paths["sft_val"], sft_val)
    write_jsonl(paths["rl_train"], rl_train)
    write_jsonl(paths["rl_val"], rl_val)
    write_jsonl(paths["verl_train_jsonl"], verl_train)
    write_jsonl(paths["verl_val_jsonl"], verl_val)
    write_json(paths["dataset_info"], _llamafactory_dataset_info())

    parquet_status: dict[str, Any] = {"requested": write_parquet, "written": False, "paths": {}}
    if write_parquet:
        # parquet 是可选能力，避免把 pyarrow 强制带进最小安装环境。
        parquet_paths = {
            "verl_train_parquet": output_dir / "verl_train.parquet",
            "verl_val_parquet": output_dir / "verl_val.parquet",
        }
        write_verl_parquet(parquet_paths["verl_train_parquet"], verl_train)
        write_verl_parquet(parquet_paths["verl_val_parquet"], verl_val)
        paths.update(parquet_paths)
        parquet_status = {
            "requested": True,
            "written": True,
            "paths": {key: str(path) for key, path in parquet_paths.items()},
        }

    summary = {
        "version": 1,
        "workspace_id": workspace_id,
        "source_name": source_name,
        "suite_id": suite_id,
        "input_path": str(input_path),
        "val_input_path": str(val_input_path) if val_input_path else None,
        "counts": {
            "train": len(train_examples),
            "val": len(val_examples),
            "task_samples": len(task_records),
            "eval_items": len(eval_records),
        },
        "outputs": {key: str(path) for key, path in paths.items()},
        "parquet": parquet_status,
    }
    write_json(paths["manifest"], summary)
    return summary


def extract_gsm8k_answer(text: str) -> str:
    marker_match = re.search(r"####\s*(.+)\s*$", text, flags=re.DOTALL)
    if marker_match:
        return normalize_math_answer(marker_match.group(1))
    boxed_match = re.search(r"\\boxed\{([^{}]+)\}", text)
    if boxed_match:
        return normalize_math_answer(boxed_match.group(1))
    number_matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if number_matches:
        return normalize_math_answer(number_matches[-1])
    return normalize_math_answer(text)


def normalize_math_answer(value: Any) -> str:
    text = str(value).strip()
    text = text.replace(",", "")
    text = text.strip().rstrip(".")
    text = re.sub(r"\s+", " ", text)
    try:
        number = Decimal(text)
    except (InvalidOperation, ValueError):
        return text
    normalized = format(number.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized or "0"


def write_verl_parquet(path: Path, records: list[dict[str, Any]]) -> None:
    try:
        import pyarrow as pa  # type: ignore[import-not-found]
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("parquet output needs pyarrow. Run: python -m pip install -e .[parquet]") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records)
    pq.write_table(table, path)


def _load_splits(
    input_path: Path,
    *,
    val_input_path: Path | None,
    train_ratio: float,
    limit: int | None,
    val_limit: int | None,
) -> tuple[list[Gsm8kExample], list[Gsm8kExample]]:
    # 用户提供独立验证集时保持原始边界，否则按固定比例切分，便于 tiny 数据离线冒烟。
    train_records = read_jsonl(input_path)
    if val_input_path is not None:
        val_records = read_jsonl(val_input_path)
        return _examples_from_records(train_records[:limit]), _examples_from_records(val_records[:val_limit])

    examples = _examples_from_records(train_records[:limit])
    if len(examples) <= 1:
        return examples, examples
    train_count = int(round(len(examples) * train_ratio))
    train_count = max(1, min(train_count, len(examples) - 1))
    return examples[:train_count], examples[train_count:]


def _examples_from_records(records: list[dict[str, Any]]) -> list[Gsm8kExample]:
    examples: list[Gsm8kExample] = []
    for index, record in enumerate(records, start=1):
        question = _extract_question(record)
        solution = _extract_solution(record)
        answer = _extract_expected_answer(record, solution)
        if not question:
            raise ValueError(f"record {index} is missing question")
        if not solution:
            raise ValueError(f"record {index} is missing answer")
        if not answer:
            raise ValueError(f"record {index} does not contain a final answer")
        raw_id = record.get("id") or record.get("sample_id") or record.get("question_id")
        examples.append(
            Gsm8kExample(
                question=question,
                solution=_solution_with_marker(solution, answer),
                answer=answer,
                raw_id=str(raw_id) if raw_id not in (None, "") else None,
            )
        )
    return examples


def _extract_question(record: dict[str, Any]) -> str:
    for key in ("question", "query", "prompt", "problem"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    input_value = record.get("input")
    if isinstance(input_value, dict):
        query = input_value.get("query")
        if isinstance(query, str) and query.strip():
            return query.strip()
    return ""


def _extract_solution(record: dict[str, Any]) -> str:
    for key in ("answer", "solution", "response", "completion"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    expected = record.get("expected")
    if isinstance(expected, dict):
        solution = expected.get("solution") or expected.get("answer")
        if isinstance(solution, str) and solution.strip():
            return solution.strip()
    return ""


def _extract_expected_answer(record: dict[str, Any], solution: str) -> str:
    expected = record.get("expected")
    if isinstance(expected, dict):
        answer = expected.get("answer")
        if answer not in (None, ""):
            return normalize_math_answer(answer)
    for key in ("final_answer", "label", "target"):
        value = record.get(key)
        if value not in (None, ""):
            return normalize_math_answer(value)
    return extract_gsm8k_answer(solution)


def _solution_with_marker(solution: str, answer: str) -> str:
    if "####" in solution:
        return solution
    return f"{solution}\n\n#### {answer}"


def _task_records(
    examples: list[Gsm8kExample],
    *,
    split: str,
    workspace_id: str,
    source_name: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, example in enumerate(examples, start=1):
        sample_id = _sample_id(split, index)
        records.append(
            {
                "sample_id": sample_id,
                "workspace_id": workspace_id,
                "input": {"query": example.question},
                "expected": {
                    "success_criteria": ["final_answer_after_hashes", "numeric_answer_exact_match"],
                    "answer": example.answer,
                    "solution": example.solution,
                },
                "tags": ["math", "gsm8k", split],
                "lineage": {"source": source_name, "split": split, "raw_id": example.raw_id},
            }
        )
    return records


def _eval_records(examples: list[Gsm8kExample], *, suite_id: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, example in enumerate(examples, start=1):
        sample_id = _sample_id("val", index)
        records.append(
            {
                "eval_item_id": f"eval-{sample_id}",
                "sample_id": sample_id,
                "suite_id": suite_id,
                "expected": {"answer": example.answer, "solution": example.solution},
                "checks": [
                    {"name": "final_answer_after_hashes", "type": "rule", "weight": 0.4},
                    {"name": "numeric_answer_exact_match", "type": "rule", "weight": 0.6},
                ],
            }
        )
    return records


def _sft_records(examples: list[Gsm8kExample], *, split: str, source_name: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, example in enumerate(examples, start=1):
        sample_id = _sample_id(split, index)
        records.append(
            {
                "id": f"sft-{sample_id}",
                "messages": [
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": example.question},
                    {"role": "assistant", "content": example.solution},
                ],
                "metadata": {
                    "sample_id": sample_id,
                    "source": source_name,
                    "split": split,
                    "answer": example.answer,
                },
            }
        )
    return records


def _rl_records(examples: list[Gsm8kExample], *, split: str, source_name: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, example in enumerate(examples, start=1):
        sample_id = _sample_id(split, index)
        records.append(
            {
                "id": f"rl-{sample_id}",
                "prompt": [{"role": "user", "content": _prompt_content(example.question)}],
                "reward_model": {"style": "rule", "ground_truth": example.answer},
                "metadata": {
                    "sample_id": sample_id,
                    "source": source_name,
                    "split": split,
                    "ability": "math",
                },
            }
        )
    return records


def _verl_records(examples: list[Gsm8kExample], *, split: str, source_name: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, example in enumerate(examples, start=1):
        sample_id = _sample_id(split, index)
        records.append(
            {
                "data_source": source_name,
                "prompt": [{"role": "user", "content": _prompt_content(example.question)}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": example.answer},
                "extra_info": {
                    "sample_id": sample_id,
                    "split": split,
                    "index": index,
                    "answer": example.answer,
                    "question": example.question,
                },
            }
        )
    return records


def _prompt_content(question: str) -> str:
    stripped = question.strip()
    if "####" in stripped:
        return stripped
    return f"{stripped}\n\n{DEFAULT_PROMPT_SUFFIX}"


def _llamafactory_dataset_info() -> dict[str, Any]:
    """返回可直接放入 LLaMA-Factory dataset_dir 的数据注册信息。"""
    return {
        "gsm8k_sft": {
            "file_name": "sft_train.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
    }


def _sample_id(split: str, index: int) -> str:
    return f"gsm8k-{split}-{index:06d}"
