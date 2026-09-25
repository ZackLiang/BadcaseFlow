from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from badcaseflow.cli import main
from badcaseflow.datasets import build_dataset_version
from badcaseflow.registry import get_dataset, list_datasets, register_dataset


ROOT = Path(__file__).resolve().parents[1]


class DatasetVersionTest(unittest.TestCase):
    def test_build_dataset_version_for_task_jsonl(self) -> None:
        seed = ROOT / "examples" / "datasets" / "seed_tasks.jsonl"

        manifest = build_dataset_version(seed, kind="task", dataset_id="demo-seed", workspace_id="demo-agent")

        self.assertTrue(manifest["dataset_version_id"].startswith("dataset-demo-seed-"))
        self.assertEqual(manifest["kind"], "task")
        self.assertEqual(manifest["record_count"], 3)
        self.assertEqual(manifest["schema_status"], "passed")
        self.assertEqual(len(manifest["sha256"]), 64)
        self.assertEqual(manifest["workspace_id"], "demo-agent")
        self.assertIn("task-001", manifest["sample_ids"])

    def test_schema_errors_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad_preference.jsonl"
            path.write_text(json.dumps({"prompt": "p", "chosen": "c"}, ensure_ascii=False) + "\n", encoding="utf-8")

            manifest = build_dataset_version(path, kind="preference")

            self.assertEqual(manifest["schema_status"], "failed")
            self.assertEqual(manifest["record_count"], 1)
            self.assertIn("rejected must be a non-empty string", manifest["schema_errors"][0]["errors"])

    def test_register_dataset_and_cli_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed = ROOT / "examples" / "datasets" / "seed_tasks.jsonl"
            output = io.StringIO()

            with redirect_stdout(output):
                self.assertEqual(
                    main(
                        [
                            "dataset",
                            "register",
                            "--root",
                            str(root),
                            "--path",
                            str(seed),
                            "--kind",
                            "task",
                            "--dataset-id",
                            "seed-tasks",
                            "--workspace",
                            "demo-agent",
                        ]
                    ),
                    0,
                )

            self.assertIn("dataset_version_id=dataset-seed-tasks-", output.getvalue())
            datasets = list_datasets(root)
            self.assertEqual(len(datasets), 1)
            dataset_version_id = datasets[0]["dataset_version_id"]
            self.assertEqual(get_dataset(root, dataset_version_id)["dataset_id"], "seed-tasks")
            self.assertTrue((root / ".badcaseflow" / "datasets" / f"{dataset_version_id}.json").exists())

            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["dataset", "list", "--root", str(root)]), 0)
            self.assertIn("seed-tasks", output.getvalue())

            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    main(["dataset", "show", "--root", str(root), "--dataset-version-id", dataset_version_id]),
                    0,
                )
            self.assertIn('"dataset_id": "seed-tasks"', output.getvalue())

    def test_register_dataset_function(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = ROOT / "examples" / "datasets" / "eval_tasks.jsonl"

            manifest = register_dataset(root, path=suite, kind="eval_suite", dataset_id="eval-suite")

            self.assertEqual(manifest["schema_status"], "passed")
            self.assertEqual(manifest["record_count"], 3)
            self.assertEqual(list_datasets(root)[0]["dataset_version_id"], manifest["dataset_version_id"])

    def test_train_recipe_resolves_dataset_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed = ROOT / "examples" / "datasets" / "seed_tasks.jsonl"
            dataset = register_dataset(root, path=seed, kind="task", dataset_id="seed-tasks")
            recipe = root / "sft_dataset_version.yaml"
            recipe.write_text(
                "\n".join(
                    [
                        "recipe_id: dataset-version-sft",
                        "backend: verl",
                        "stage: sft",
                        "model:",
                        "  path: Qwen/Qwen3-0.6B",
                        "data:",
                        f"  dataset_version_id: {dataset['dataset_version_id']}",
                        "training:",
                        "  train_batch_size: 2",
                        "outputs:",
                        "  output_dir: artifacts/demo-agent/dataset-version-sft",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    main(
                        [
                            "train",
                            "dry-run",
                            "--recipe",
                            str(recipe),
                            "--registry-root",
                            str(root),
                            "--output-root",
                            str(root / "runs"),
                        ]
                    ),
                    0,
                )

            self.assertIn("datasets=1", output.getvalue())
            manifest_path = root / "runs" / "dry_runs" / "dry-run-dataset-version-sft.json"
            dry_run = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(dry_run["dataset_resolution"]["datasets"][0]["dataset_id"], "seed-tasks")
            self.assertTrue(any("data.train_files" in part and "seed_tasks.jsonl" in part for part in dry_run["command"]))

    def test_eval_recipe_resolves_trace_and_suite_versions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            traces = ROOT / "examples" / "datasets" / "external_traces.jsonl"
            suite = ROOT / "examples" / "datasets" / "eval_tasks.jsonl"
            trace_dataset = register_dataset(root, path=traces, kind="trace", dataset_id="external-traces")
            suite_dataset = register_dataset(root, path=suite, kind="eval_suite", dataset_id="eval-suite")
            recipe = root / "builtin_eval_dataset_version.yaml"
            recipe.write_text(
                "\n".join(
                    [
                        "recipe_id: dataset-version-eval",
                        "backend: builtin",
                        "stage: agent_trace",
                        "data:",
                        f"  trace_dataset_version_id: {trace_dataset['dataset_version_id']}",
                        f"  suite_dataset_version_id: {suite_dataset['dataset_version_id']}",
                        "outputs:",
                        "  output_dir: runs/eval_runs/dataset-version-eval",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    main(
                        [
                            "evals",
                            "dry-run",
                            "--recipe",
                            str(recipe),
                            "--registry-root",
                            str(root),
                            "--output-root",
                            str(root / "runs"),
                        ]
                    ),
                    0,
                )

            self.assertIn("datasets=2", output.getvalue())
            manifest_path = root / "runs" / "dry_runs" / "dry-run-dataset-version-eval.json"
            dry_run = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(dry_run["execution"]["traces"], str(traces))
            self.assertEqual(dry_run["execution"]["suite"], str(suite))
            self.assertEqual(len(dry_run["dataset_resolution"]["datasets"]), 2)


if __name__ == "__main__":
    unittest.main()
