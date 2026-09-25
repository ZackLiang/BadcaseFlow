from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from badcaseflow.cli import main


ROOT = Path(__file__).resolve().parents[1]


class DeploymentPlanTest(unittest.TestCase):
    def test_deploy_plan_registers_approved_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry_root = root / "registry"
            output = root / "deploy" / "plan.json"
            recipe = ROOT / "examples" / "recipes" / "command_smoke.example.yaml"
            train_run = root / "runs" / "train_runs" / "deploy-train"
            eval_report = root / "eval_report.json"
            eval_report.write_text(
                json.dumps(
                    {
                        "suite_id": "deploy-smoke",
                        "total": 3,
                        "passed": 3,
                        "failed": 0,
                        "pass_rate": 1.0,
                        "failure_counts": {},
                    },
                    ensure_ascii=False,
                )
                + "\n",
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
                        "deploy-train",
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "registry",
                        "register-model",
                        "--root",
                        str(registry_root),
                        "--run",
                        str(train_run),
                        "--model-id",
                        "deploy-agent",
                        "--model-path",
                        "artifacts/demo-agent/deploy-agent",
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "registry",
                        "attach-eval",
                        "--root",
                        str(registry_root),
                        "--model-id",
                        "deploy-agent",
                        "--eval-report",
                        str(eval_report),
                    ]
                ),
                0,
            )

            output_text = io.StringIO()
            with redirect_stdout(output_text):
                self.assertEqual(
                    main(
                        [
                            "deploy",
                            "plan",
                            "--root",
                            str(registry_root),
                            "--model-id",
                            "deploy-agent",
                            "--backend",
                            "vllm",
                            "--deployment-id",
                            "deploy-agent-v1",
                            "--canary-percent",
                            "10",
                            "--output",
                            str(output),
                            "--register",
                        ]
                    ),
                    0,
                )

            self.assertIn("deployment_id=deploy-agent-v1", output_text.getvalue())
            plan = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(plan["backend"], "vllm")
            self.assertEqual(plan["canary"]["percent"], 10.0)
            self.assertIn("vllm.entrypoints.openai.api_server", plan["commands"]["serve"][0])
            self.assertTrue(output.with_suffix(".ps1").exists())

            registry = json.loads((registry_root / ".badcaseflow" / "registry.json").read_text(encoding="utf-8"))
            self.assertEqual(registry["deployments"][0]["deployment_id"], "deploy-agent-v1")
            self.assertEqual(registry["models"][0]["latest_deployment_id"], "deploy-agent-v1")

            output_text = io.StringIO()
            with redirect_stdout(output_text):
                self.assertEqual(main(["deploy", "list", "--root", str(registry_root)]), 0)
            self.assertIn("deploy-agent-v1", output_text.getvalue())

            output_text = io.StringIO()
            with redirect_stdout(output_text):
                self.assertEqual(
                    main(["deploy", "show", "--root", str(registry_root), "--deployment-id", "deploy-agent-v1"]),
                    0,
                )
            self.assertIn('"model_id": "deploy-agent"', output_text.getvalue())

            output_text = io.StringIO()
            with redirect_stdout(output_text):
                self.assertEqual(
                    main(["registry", "model-lineage", "--root", str(registry_root), "--model-id", "deploy-agent"]),
                    0,
                )
            lineage = json.loads(output_text.getvalue())
            self.assertEqual(lineage["deployments"][0]["deployment_id"], "deploy-agent-v1")

    def test_deploy_plan_requires_approved_model_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry_root = root / "registry"
            recipe = ROOT / "examples" / "recipes" / "command_smoke.example.yaml"
            train_run = root / "runs" / "train_runs" / "blocked-train"

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
                        "blocked-train",
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "registry",
                        "register-model",
                        "--root",
                        str(registry_root),
                        "--run",
                        str(train_run),
                        "--model-id",
                        "blocked-agent",
                    ]
                ),
                0,
            )
            with redirect_stderr(io.StringIO()):
                self.assertEqual(
                    main(["deploy", "plan", "--root", str(registry_root), "--model-id", "blocked-agent"]),
                    1,
                )


if __name__ == "__main__":
    unittest.main()
