from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from badcaseflow.cli import main
from badcaseflow.evaluation import evaluate_traces
from badcaseflow.gsm8k import extract_gsm8k_answer, prepare_gsm8k
from badcaseflow.io_utils import read_jsonl


ROOT = Path(__file__).resolve().parents[1]


class Gsm8kPrepareTest(unittest.TestCase):
    def test_extract_answer_from_gsm8k_solution(self) -> None:
        self.assertEqual(extract_gsm8k_answer("We compute 20 + 3 = 23. #### 23"), "23")
        self.assertEqual(extract_gsm8k_answer("Final value is 1,200. #### 1,200"), "1200")
        self.assertEqual(extract_gsm8k_answer("The answer is 10.0"), "10")

    def test_prepare_gsm8k_outputs_training_and_eval_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "gsm8k"
            summary = prepare_gsm8k(
                ROOT / "examples" / "datasets" / "gsm8k_tiny.jsonl",
                output_dir,
            )

            self.assertEqual(summary["counts"]["train"], 4)
            self.assertEqual(summary["counts"]["val"], 1)
            self.assertEqual(summary["counts"]["task_samples"], 5)
            self.assertEqual(summary["counts"]["eval_items"], 1)

            tasks = read_jsonl(output_dir / "task_samples.jsonl")
            suite = read_jsonl(output_dir / "eval_suite.jsonl")
            sft = read_jsonl(output_dir / "sft_train.jsonl")
            rl = read_jsonl(output_dir / "rl_train.jsonl")
            verl = read_jsonl(output_dir / "verl_train.jsonl")
            dataset_info = json.loads((output_dir / "dataset_info.json").read_text(encoding="utf-8"))

            self.assertEqual(tasks[0]["sample_id"], "gsm8k-train-000001")
            self.assertEqual(tasks[0]["expected"]["answer"], "23")
            self.assertEqual(suite[0]["sample_id"], "gsm8k-val-000001")
            self.assertEqual(suite[0]["expected"]["answer"], "17")
            self.assertEqual(sft[0]["messages"][-1]["role"], "assistant")
            self.assertEqual(rl[0]["reward_model"]["ground_truth"], "23")
            self.assertEqual(verl[0]["ability"], "math")
            self.assertEqual(dataset_info["gsm8k_sft"]["formatting"], "sharegpt")
            self.assertTrue((output_dir / "manifest.json").exists())

    def test_cli_prepare_gsm8k_can_register_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(
                    [
                        "data",
                        "prepare-gsm8k",
                        "--input",
                        str(ROOT / "examples" / "datasets" / "gsm8k_tiny.jsonl"),
                        "--output-dir",
                        str(root / "runs" / "gsm8k"),
                        "--registry-root",
                        str(root),
                        "--register",
                    ]
                )
            self.assertEqual(code, 0)
            text = output.getvalue()
            self.assertIn("prepared_gsm8k=", text)
            self.assertIn("train=4 val=1", text)
            self.assertIn("dataset_id=gsm8k-sft-train", text)

            registry = json.loads((root / ".badcaseflow" / "registry.json").read_text(encoding="utf-8"))
            dataset_ids = {item["dataset_id"] for item in registry["datasets"]}
            self.assertIn("gsm8k-task-samples", dataset_ids)
            self.assertIn("gsm8k-verl-train", dataset_ids)

    def test_gsm8k_eval_checks_numeric_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "gsm8k"
            prepare_gsm8k(ROOT / "examples" / "datasets" / "gsm8k_tiny.jsonl", output_dir)
            suite = read_jsonl(output_dir / "eval_suite.jsonl")
            trace = {
                "trace_id": "trace-gsm8k-val-000001",
                "sample_id": "gsm8k-val-000001",
                "workspace_id": "gsm8k-agent",
                "steps": [],
                "final_answer": "Nora spends 17 dollars. #### 17",
            }

            results, report, accepted, rejected = evaluate_traces([trace], suite)

            self.assertTrue(results[0]["passed"])
            self.assertEqual(report["pass_rate"], 1.0)
            self.assertEqual(len(accepted), 1)
            self.assertEqual(len(rejected), 0)

            bad_trace = dict(trace)
            bad_trace["final_answer"] = "Nora spends 18 dollars. #### 18"
            results, report, accepted, rejected = evaluate_traces([bad_trace], suite)
            self.assertFalse(results[0]["passed"])
            self.assertEqual(report["pass_rate"], 0.0)
            self.assertEqual(len(accepted), 0)
            self.assertEqual(len(rejected), 1)


if __name__ == "__main__":
    unittest.main()
