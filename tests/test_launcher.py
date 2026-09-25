from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from badcaseflow.cli import main


ROOT = Path(__file__).resolve().parents[1]


class LauncherPlanTest(unittest.TestCase):
    def test_local_train_launch_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "launch_train.json"
            recipe = ROOT / "examples" / "recipes" / "command_smoke.example.yaml"
            self.assertEqual(
                main(
                    [
                        "launcher",
                        "plan",
                        "--kind",
                        "train",
                        "--recipe",
                        str(recipe),
                        "--launcher",
                        "local",
                        "--mode",
                        "run",
                        "--workspace",
                        "demo-agent",
                        "--run-id",
                        "launcher-train",
                        "--output-root",
                        str(root / "runs"),
                        "--output",
                        str(output),
                    ]
                ),
                0,
            )
            plan = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(plan["kind"], "train")
            self.assertEqual(plan["launcher"], "local")
            self.assertIn("train run", plan["commands"]["run"][0])
            self.assertIn("--run-id launcher-train", plan["commands"]["run"][0])
            self.assertTrue(output.with_suffix(".ps1").exists())

    def test_ssh_eval_launch_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / ".badcaseflow" / "remotes.json"
            output = root / "launch_eval.json"
            recipe = ROOT / "examples" / "evals" / "command_eval_smoke.example.yaml"
            self.assertEqual(
                main(
                    [
                        "remote",
                        "add",
                        "--name",
                        "autodl",
                        "--host",
                        "root@example.autodl",
                        "--workdir",
                        "/root/BadcaseFlow",
                        "--config",
                        str(config),
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "launcher",
                        "plan",
                        "--kind",
                        "eval",
                        "--recipe",
                        str(recipe),
                        "--launcher",
                        "ssh",
                        "--target",
                        "autodl",
                        "--config",
                        str(config),
                        "--mode",
                        "plan",
                        "--run-id",
                        "launcher-eval",
                        "--output-root",
                        str(root / "runs"),
                        "--output",
                        str(output),
                    ]
                ),
                0,
            )
            plan = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(plan["kind"], "eval")
            self.assertEqual(plan["launcher"], "ssh")
            self.assertEqual(plan["target"], "autodl")
            self.assertIn("evals run", plan["commands"]["run"][0])
            self.assertIn("--plan-only", plan["commands"]["run"][0])
            self.assertIn("/root/BadcaseFlow/recipes/command_eval_smoke.example.yaml", plan["commands"]["run"][0])
            self.assertTrue(output.with_suffix(".ps1").exists())


if __name__ == "__main__":
    unittest.main()
