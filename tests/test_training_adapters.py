from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from badcaseflow.cli import main
from badcaseflow.recipes import build_train_dry_run
from badcaseflow.training import list_training_adapters
from badcaseflow.training.harness import parse_metric_line


ROOT = Path(__file__).resolve().parents[1]


class TrainingAdapterTest(unittest.TestCase):
    def test_list_training_adapters_has_multiple_framework_stages(self) -> None:
        rows = list_training_adapters()
        pairs = {(row["backend"], row["stage"]) for row in rows}
        self.assertIn(("llamafactory", "sft"), pairs)
        self.assertIn(("llamafactory", "dpo"), pairs)
        self.assertIn(("llamafactory", "kto"), pairs)
        self.assertIn(("verl", "sft"), pairs)
        self.assertIn(("verl", "grpo"), pairs)
        self.assertIn(("verl", "opd"), pairs)
        self.assertIn(("command", "custom"), pairs)

    def test_llamafactory_sft_and_dpo_commands(self) -> None:
        sft_recipe = ROOT / "examples" / "recipes" / "sft_llamafactory.example.yaml"
        dpo_recipe = ROOT / "examples" / "recipes" / "dpo_llamafactory.example.yaml"

        sft = build_train_dry_run(sft_recipe, workspace="demo-agent")
        dpo = build_train_dry_run(dpo_recipe, workspace="demo-agent")

        self.assertEqual(sft["backend"], "llamafactory")
        self.assertEqual(sft["stage"], "sft")
        self.assertEqual(sft["command"][:2], ["llamafactory-cli", "train"])
        self.assertEqual(dpo["stage"], "dpo")
        self.assertEqual(dpo["command"][:2], ["llamafactory-cli", "train"])

    def test_llamafactory_export_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recipe = Path(tmp) / "export.yaml"
            recipe.write_text(
                "\n".join(
                    [
                        "recipe_id: export.llamafactory.demo",
                        "backend: llamafactory",
                        "stage: export",
                        "model_name_or_path: artifacts/demo-agent/sft-v0.1",
                        "output_dir: artifacts/demo-agent/exported",
                    ]
                ),
                encoding="utf-8",
            )

            manifest = build_train_dry_run(recipe)
            self.assertEqual(manifest["stage"], "export")
            self.assertEqual(manifest["command"][:2], ["llamafactory-cli", "export"])

    def test_verl_sft_grpo_and_opd_commands(self) -> None:
        sft_recipe = ROOT / "examples" / "recipes" / "sft_verl.example.yaml"
        grpo_recipe = ROOT / "examples" / "recipes" / "grpo_verl.example.yaml"
        opd_recipe = ROOT / "examples" / "recipes" / "opd_verl.example.yaml"

        sft = build_train_dry_run(sft_recipe)
        grpo = build_train_dry_run(grpo_recipe)
        opd = build_train_dry_run(opd_recipe)

        self.assertEqual(sft["command"][:3], ["python", "-m", "verl.trainer.sft_trainer"])
        self.assertIn("model.path=Qwen/Qwen3-0.6B", sft["command"])
        self.assertEqual(grpo["command"][:3], ["python", "-m", "verl.trainer.main_ppo"])
        self.assertIn("algorithm.adv_estimator=grpo", grpo["command"])
        self.assertEqual(opd["stage"], "opd")
        self.assertIn("distillation.enabled=True", opd["command"])
        self.assertIn("distillation.teacher_models.teacher_model.model_path=Qwen/Qwen3-1.7B", opd["command"])

    def test_train_dry_run_writes_stage_to_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "runs"
            recipe = ROOT / "examples" / "recipes" / "grpo_verl.example.yaml"
            self.assertEqual(
                main(
                    [
                        "train",
                        "dry-run",
                        "--workspace",
                        "demo-agent",
                        "--recipe",
                        str(recipe),
                        "--output-root",
                        str(output_root),
                    ]
                ),
                0,
            )
            manifests = list((output_root / "dry_runs").glob("*.json"))
            self.assertEqual(len(manifests), 1)
            data = json.loads(manifests[0].read_text(encoding="utf-8"))
            self.assertEqual(data["backend"], "verl")
            self.assertEqual(data["stage"], "grpo")

    def test_cli_train_adapters(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["train", "adapters"]), 0)
        text = output.getvalue()
        self.assertIn("llamafactory", text)
        self.assertIn("verl", text)
        self.assertIn("opd", text)

    def test_unknown_backend_fails_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recipe = Path(tmp) / "unknown.yaml"
            recipe.write_text(
                "\n".join(
                    [
                        "recipe_id: unknown.demo",
                        "backend: no_such_backend",
                        "stage: sft",
                    ]
                ),
                encoding="utf-8",
            )
            manifest = build_train_dry_run(recipe)
            self.assertEqual(manifest["preflight"]["status"], "failed")
            self.assertIn("not implemented", manifest["preflight"]["errors"][0])

    def test_custom_command_harness_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recipe = root / "command.yaml"
            recipe.write_text(
                "\n".join(
                    [
                        "recipe_id: command.success.demo",
                        "backend: command",
                        "stage: custom",
                        "command:",
                        "  - python",
                        "  - -c",
                        '  - "import os,sys; print(\'hello\'); print(os.environ.get(\'BCF_TEST_ENV\')); sys.stderr.write(\'warn\\\\n\')"',
                    ]
                ),
                encoding="utf-8",
            )
            output_root = root / "runs"
            self.assertEqual(
                main(
                    [
                        "train",
                        "run",
                        "--workspace",
                        "demo-agent",
                        "--recipe",
                        str(recipe),
                        "--output-root",
                        str(output_root),
                        "--run-id",
                        "command-success",
                        "--env",
                        "BCF_TEST_ENV=ok",
                    ]
                ),
                0,
            )
            run_dir = output_root / "train_runs" / "command-success"
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "succeeded")
            self.assertEqual(status["return_code"], 0)
            self.assertEqual(status["diagnosis"]["category"], "SUCCEEDED")
            self.assertIn("hello", (run_dir / "stdout.log").read_text(encoding="utf-8"))
            self.assertIn("ok", (run_dir / "stdout.log").read_text(encoding="utf-8"))
            self.assertIn("warn", (run_dir / "stderr.log").read_text(encoding="utf-8"))
            artifact_manifest = json.loads((run_dir / "artifacts.json").read_text(encoding="utf-8"))
            artifact_names = {artifact["name"] for artifact in artifact_manifest["artifacts"]}
            self.assertIn("diagnosis", artifact_names)
            self.assertIn("metrics", artifact_names)

    def test_harness_extracts_metrics_and_inspect_prints_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recipe = root / "command_metrics.yaml"
            recipe.write_text(
                "\n".join(
                    [
                        "recipe_id: command.metrics.demo",
                        "backend: command",
                        "stage: custom",
                        "command:",
                        "  - python",
                        "  - -c",
                        '  - "print(\'metric: step=1 loss=0.5 reward=1.0\'); print(\'metric: step=2 loss=0.25 kl=0.01\')"',
                    ]
                ),
                encoding="utf-8",
            )
            output_root = root / "runs"
            self.assertEqual(
                main(
                    [
                        "train",
                        "run",
                        "--recipe",
                        str(recipe),
                        "--output-root",
                        str(output_root),
                        "--run-id",
                        "command-metrics",
                    ]
                ),
                0,
            )
            run_dir = output_root / "train_runs" / "command-metrics"
            metrics = [
                json.loads(line)
                for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(metrics), 2)
            self.assertEqual(metrics[-1]["step"], 2)
            self.assertEqual(metrics[-1]["values"]["loss"], 0.25)

            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["train", "inspect", "--run", str(run_dir)]), 0)
            text = output.getvalue()
            self.assertIn("status=succeeded", text)
            self.assertIn("metrics=2", text)
            self.assertIn("loss=0.25", text)

            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["train", "list", "--root", str(output_root / "train_runs")]), 0)
            list_text = output.getvalue()
            self.assertIn("command-metrics", list_text)
            self.assertIn("succeeded", list_text)

            registry_root = root / "registry-root"
            self.assertEqual(
                main(["registry", "index-train-run", "--root", str(registry_root), "--run", str(run_dir)]),
                0,
            )
            registry_path = registry_root / ".badcaseflow" / "registry.json"
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            artifact_ids = {artifact["artifact_id"] for artifact in registry["artifacts"]}
            self.assertIn("command-metrics:metrics", artifact_ids)
            self.assertIn("command-metrics:diagnosis", artifact_ids)

            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["registry", "artifacts", "--root", str(registry_root)]), 0)
            self.assertIn("command-metrics:metrics", output.getvalue())

            self.assertEqual(
                main(
                    [
                        "registry",
                        "register-model",
                        "--root",
                        str(registry_root),
                        "--run",
                        str(run_dir),
                        "--model-id",
                        "demo-agent-candidate",
                        "--model-path",
                        "artifacts/demo-agent/candidate",
                    ]
                ),
                0,
            )
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            self.assertEqual(registry["models"][0]["model_id"], "demo-agent-candidate")
            self.assertEqual(registry["models"][0]["metrics"]["loss"], 0.25)

            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["registry", "models", "--root", str(registry_root)]), 0)
            self.assertIn("demo-agent-candidate", output.getvalue())

            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    main(["registry", "show-model", "--root", str(registry_root), "--model-id", "demo-agent-candidate"]),
                    0,
                )
            self.assertIn("artifacts/demo-agent/candidate", output.getvalue())

            eval_report_path = root / "eval_report.json"
            eval_report_path.write_text(
                json.dumps(
                    {
                        "total": 10,
                        "passed": 9,
                        "failed": 1,
                        "pass_rate": 0.9,
                        "failure_counts": {"tool_error": 1},
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                main(
                    [
                        "registry",
                        "attach-eval",
                        "--root",
                        str(registry_root),
                        "--model-id",
                        "demo-agent-candidate",
                        "--eval-report",
                        str(eval_report_path),
                        "--min-pass-rate",
                        "0.8",
                    ]
                ),
                0,
            )
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            self.assertEqual(registry["models"][0]["status"], "blocked")
            self.assertEqual(registry["evaluations"][0]["summary"]["pass_rate"], 0.9)
            self.assertEqual(registry["promotions"][0]["decision"], "blocked")
            self.assertIn("failed_cases_present", registry["promotions"][0]["reasons"])

            self.assertEqual(
                main(
                    [
                        "registry",
                        "attach-eval",
                        "--root",
                        str(registry_root),
                        "--model-id",
                        "demo-agent-candidate",
                        "--eval-report",
                        str(eval_report_path),
                        "--fail-on-blocked",
                    ]
                ),
                2,
            )

            self.assertEqual(
                main(
                    [
                        "registry",
                        "attach-eval",
                        "--root",
                        str(registry_root),
                        "--model-id",
                        "demo-agent-candidate",
                        "--eval-report",
                        str(eval_report_path),
                        "--allow-failed-cases",
                    ]
                ),
                0,
            )
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            self.assertEqual(registry["models"][0]["status"], "approved")

            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["registry", "evaluations", "--root", str(registry_root)]), 0)
            self.assertIn("demo-agent-candidate", output.getvalue())

            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["registry", "promotions", "--root", str(registry_root)]), 0)
            self.assertIn("approved", output.getvalue())
            self.assertIn("blocked", output.getvalue())

            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    main(
                        [
                            "registry",
                            "model-lineage",
                            "--root",
                            str(registry_root),
                            "--model-id",
                            "demo-agent-candidate",
                        ]
                    ),
                    0,
                )
            lineage_text = output.getvalue()
            self.assertIn("train_artifacts", lineage_text)
            lineage = json.loads(lineage_text)
            self.assertTrue(
                any(evaluation["report_path"] == str(eval_report_path) for evaluation in lineage["evaluations"])
            )

    def test_custom_command_harness_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recipe = root / "command_fail.yaml"
            recipe.write_text(
                "\n".join(
                    [
                        "recipe_id: command.failure.demo",
                        "backend: command",
                        "stage: custom",
                        "command:",
                        "  - python",
                        "  - -c",
                        '  - "import sys; print(\'before failure\'); sys.exit(3)"',
                    ]
                ),
                encoding="utf-8",
            )
            output_root = root / "runs"
            self.assertEqual(
                main(
                    [
                        "train",
                        "run",
                        "--recipe",
                        str(recipe),
                        "--output-root",
                        str(output_root),
                        "--run-id",
                        "command-failure",
                    ]
                ),
                2,
            )
            status = json.loads(
                (output_root / "train_runs" / "command-failure" / "status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["return_code"], 3)
            self.assertEqual(status["diagnosis"]["category"], "PROCESS_FAILED")

    def test_train_failure_diagnosis_for_missing_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recipe = root / "missing_dependency.yaml"
            recipe.write_text(
                "\n".join(
                    [
                        "recipe_id: command.missing_dependency.demo",
                        "backend: command",
                        "stage: custom",
                        "command:",
                        "  - python",
                        "  - -c",
                        "  - |",
                        "    import sys; sys.stderr.write(\"ModuleNotFoundError: No module named 'verl'\\n\"); sys.exit(1)",
                    ]
                ),
                encoding="utf-8",
            )
            output_root = root / "runs"
            self.assertEqual(
                main(
                    [
                        "train",
                        "run",
                        "--recipe",
                        str(recipe),
                        "--output-root",
                        str(output_root),
                        "--run-id",
                        "missing-dependency",
                    ]
                ),
                2,
            )
            run_dir = output_root / "train_runs" / "missing-dependency"
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            diagnosis = json.loads((run_dir / "diagnosis.json").read_text(encoding="utf-8"))
            self.assertEqual(status["diagnosis"]["category"], "DEPENDENCY_MISSING")
            self.assertEqual(diagnosis["category"], "DEPENDENCY_MISSING")

            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["train", "inspect", "--run", str(run_dir)]), 2)
            self.assertIn("diagnosis=DEPENDENCY_MISSING", output.getvalue())

    def test_train_failure_diagnosis_for_cuda_oom(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recipe = root / "cuda_oom.yaml"
            recipe.write_text(
                "\n".join(
                    [
                        "recipe_id: command.cuda_oom.demo",
                        "backend: command",
                        "stage: custom",
                        "command:",
                        "  - python",
                        "  - -c",
                        "  - |",
                        "    import sys; sys.stderr.write('torch.cuda.OutOfMemoryError: CUDA out of memory\\n'); sys.exit(1)",
                    ]
                ),
                encoding="utf-8",
            )
            output_root = root / "runs"
            self.assertEqual(
                main(
                    [
                        "train",
                        "run",
                        "--recipe",
                        str(recipe),
                        "--output-root",
                        str(output_root),
                        "--run-id",
                        "cuda-oom",
                    ]
                ),
                2,
            )
            status = json.loads((output_root / "train_runs" / "cuda-oom" / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["diagnosis"]["category"], "CUDA_OOM")

    def test_train_run_plan_only_writes_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recipe = root / "command_plan.yaml"
            recipe.write_text(
                "\n".join(
                    [
                        "recipe_id: command.plan.demo",
                        "backend: command",
                        "stage: custom",
                        "command:",
                        "  - python",
                        "  - -c",
                        '  - "print(\'planned only\')"',
                    ]
                ),
                encoding="utf-8",
            )
            output_root = root / "runs"
            self.assertEqual(
                main(
                    [
                        "train",
                        "run",
                        "--recipe",
                        str(recipe),
                        "--output-root",
                        str(output_root),
                        "--run-id",
                        "command-plan",
                        "--plan-only",
                    ]
                ),
                0,
            )
            run_dir = output_root / "train_runs" / "command-plan"
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "planned")
            self.assertEqual(status["diagnosis"]["category"], "PLANNED")
            self.assertTrue((run_dir / "manifest.json").exists())
            self.assertTrue((run_dir / "events.jsonl").exists())
            self.assertTrue((run_dir / "artifacts.json").exists())

    def test_train_run_blocks_failed_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recipe = root / "invalid_command.yaml"
            recipe.write_text(
                "\n".join(
                    [
                        "recipe_id: invalid.command.demo",
                        "backend: command",
                        "stage: custom",
                    ]
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                main(
                    [
                        "train",
                        "run",
                        "--recipe",
                        str(recipe),
                        "--output-root",
                        str(root / "runs"),
                        "--run-id",
                        "should-not-run",
                    ]
                ),
                2,
            )
            self.assertFalse((root / "runs" / "train_runs" / "should-not-run").exists())

    def test_metric_parser_ignores_plain_logs(self) -> None:
        self.assertIsNone(parse_metric_line("hello world"))
        metric = parse_metric_line("metric: step=10 train/loss=1.25e-1 reward=2")
        self.assertIsNotNone(metric)
        assert metric is not None
        self.assertEqual(metric["step"], 10)
        self.assertEqual(metric["values"]["train/loss"], 0.125)
        self.assertEqual(metric["values"]["reward"], 2)


if __name__ == "__main__":
    unittest.main()
