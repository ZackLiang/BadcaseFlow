from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from badcaseflow.cli import main


ROOT = Path(__file__).resolve().parents[1]


class FlywheelRoundTest(unittest.TestCase):
    def test_round_run_local_writes_lineage_and_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            round_dir = root / "runs" / "rounds" / "round-smoke"
            registry_root = root / "registry"
            seed = ROOT / "examples" / "datasets" / "seed_tasks.jsonl"
            suite = ROOT / "examples" / "datasets" / "eval_tasks.jsonl"
            recipe = ROOT / "examples" / "recipes" / "command_smoke.example.yaml"

            self.assertEqual(
                main(
                    [
                        "round",
                        "run-local",
                        "--workspace",
                        "demo-agent",
                        "--round",
                        str(round_dir),
                        "--seed",
                        str(seed),
                        "--suite",
                        str(suite),
                        "--recipe",
                        str(recipe),
                        "--train-mode",
                        "run",
                        "--run-id",
                        "round-command-smoke",
                        "--registry-root",
                        str(registry_root),
                        "--model-id",
                        "round-agent-candidate",
                        "--attach-round-eval",
                    ]
                ),
                0,
            )

            manifest = json.loads((round_dir / "round.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "succeeded")
            self.assertEqual(manifest["quality_status"], "blocked")
            self.assertEqual(manifest["summary"]["model_id"], "round-agent-candidate")
            self.assertEqual(manifest["summary"]["promotion_decision"], "blocked")
            self.assertTrue((round_dir / "events.jsonl").exists())
            self.assertTrue((round_dir / "data_run" / "sft.jsonl").exists())
            self.assertTrue((round_dir / "data_run" / "dataset_versions.json").exists())
            self.assertTrue((round_dir / "iteration_plan.json").exists())
            self.assertTrue((round_dir / "iteration_plan.md").exists())
            self.assertTrue((round_dir / "train_runs" / "round-command-smoke" / "status.json").exists())
            self.assertEqual(manifest["datasets"]["sft"]["schema_status"], "passed")
            self.assertEqual(manifest["datasets"]["tasks"]["record_count"], 3)
            self.assertEqual(
                [stage["name"] for stage in manifest["stages"]],
                ["ingest", "rollout", "eval", "analyze", "export", "train", "registry", "plan_next"],
            )
            dataset_versions = json.loads((round_dir / "data_run" / "dataset_versions.json").read_text(encoding="utf-8"))
            self.assertEqual(dataset_versions["summary"]["total"], 9)
            self.assertEqual(dataset_versions["summary"]["failed"], 0)
            plan = json.loads((round_dir / "iteration_plan.json").read_text(encoding="utf-8"))
            action_types = {action["action_type"] for action in plan["actions"]}
            self.assertIn("BLOCK_PROMOTION", action_types)
            self.assertIn("PATCH_WORKFLOW", action_types)

            registry = json.loads((registry_root / ".badcaseflow" / "registry.json").read_text(encoding="utf-8"))
            self.assertEqual(registry["models"][0]["model_id"], "round-agent-candidate")
            self.assertEqual(registry["models"][0]["status"], "blocked")
            self.assertEqual(len(registry["models"][0]["dataset_version_ids"]), 9)
            self.assertEqual(registry["promotions"][0]["decision"], "blocked")
            self.assertIn("failed_cases_present", registry["promotions"][0]["reasons"])
            self.assertEqual(len(registry["datasets"]), 9)
            self.assertEqual(registry["datasets"][0]["schema_status"], "passed")

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
                            "round-agent-candidate",
                        ]
                    ),
                    0,
                )
            lineage = json.loads(output.getvalue())
            self.assertEqual(len(lineage["datasets"]), 9)

            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["round", "inspect", "--round", str(round_dir)]), 0)
            self.assertIn("quality_status=blocked", output.getvalue())
            self.assertIn("stage=registry status=succeeded", output.getvalue())

            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["round", "list", "--root", str(root / "runs" / "rounds")]), 0)
            self.assertIn("round-smoke", output.getvalue())

            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["round", "plan-next", "--round", str(round_dir)]), 0)
            self.assertIn("actions=2", output.getvalue())

    def test_round_can_run_candidate_eval_and_promote_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            round_dir = root / "runs" / "rounds" / "round-candidate-eval"
            registry_root = root / "registry"
            seed = ROOT / "examples" / "datasets" / "seed_tasks.jsonl"
            suite = ROOT / "examples" / "datasets" / "eval_tasks.jsonl"
            traces = ROOT / "examples" / "datasets" / "external_traces.jsonl"
            recipe = ROOT / "examples" / "recipes" / "command_smoke.example.yaml"
            eval_recipe = root / "candidate_eval.yaml"
            eval_recipe.write_text(
                "\n".join(
                    [
                        "recipe_id: candidate-eval",
                        "backend: builtin",
                        "stage: agent_trace",
                        "data:",
                        f"  traces: {traces}",
                        f"  suite: {suite}",
                        "gate:",
                        "  min_pass_rate: 0.8",
                        "outputs:",
                        "  output_dir: runs/eval_runs/candidate-eval",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            self.assertEqual(
                main(
                    [
                        "round",
                        "run-local",
                        "--workspace",
                        "demo-agent",
                        "--round",
                        str(round_dir),
                        "--seed",
                        str(seed),
                        "--suite",
                        str(suite),
                        "--recipe",
                        str(recipe),
                        "--train-mode",
                        "run",
                        "--run-id",
                        "round-candidate-train",
                        "--registry-root",
                        str(registry_root),
                        "--model-id",
                        "candidate-eval-agent",
                        "--candidate-eval-recipe",
                        str(eval_recipe),
                        "--candidate-eval-run-id",
                        "candidate-eval-run",
                    ]
                ),
                0,
            )

            manifest = json.loads((round_dir / "round.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "succeeded")
            self.assertEqual(manifest["quality_status"], "approved")
            self.assertEqual(manifest["summary"]["initial_failed"], 1)
            self.assertEqual(manifest["summary"]["failed"], 0)
            self.assertEqual(manifest["summary"]["candidate_eval_run_id"], "candidate-eval-run")
            self.assertEqual(
                [stage["name"] for stage in manifest["stages"]],
                [
                    "ingest",
                    "rollout",
                    "eval",
                    "analyze",
                    "export",
                    "train",
                    "candidate_eval",
                    "candidate_analyze",
                    "registry",
                    "plan_next",
                ],
            )
            self.assertTrue((round_dir / "eval_runs" / "candidate-eval-run" / "eval_report.json").exists())
            self.assertTrue((round_dir / "eval_runs" / "candidate-eval-run" / "case_report.json").exists())

            registry = json.loads((registry_root / ".badcaseflow" / "registry.json").read_text(encoding="utf-8"))
            self.assertEqual(registry["models"][0]["model_id"], "candidate-eval-agent")
            self.assertEqual(registry["models"][0]["status"], "approved")
            self.assertEqual(registry["evaluations"][0]["eval_run_id"], "candidate-eval-run")
            self.assertEqual(registry["promotions"][0]["decision"], "approved")

            plan = json.loads((round_dir / "iteration_plan.json").read_text(encoding="utf-8"))
            action_types = {action["action_type"] for action in plan["actions"]}
            self.assertEqual(plan["summary"]["failed"], 0)
            self.assertIn("PROMOTE_CANDIDATE", action_types)


if __name__ == "__main__":
    unittest.main()
