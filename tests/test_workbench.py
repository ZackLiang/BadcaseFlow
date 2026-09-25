from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from badcaseflow.cli import main


ROOT = Path(__file__).resolve().parents[1]


class WorkbenchExportTest(unittest.TestCase):
    def test_workbench_build_exports_html_and_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            round_dir = root / "runs" / "rounds" / "round-smoke"
            registry_root = root / "registry"
            output = root / "runs" / "workbench" / "index.html"
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
                        "workbench-train",
                        "--registry-root",
                        str(registry_root),
                        "--model-id",
                        "workbench-agent",
                        "--attach-round-eval",
                    ]
                ),
                0,
            )

            output_text = io.StringIO()
            with redirect_stdout(output_text):
                self.assertEqual(
                    main(
                        [
                            "workbench",
                            "build",
                            "--root",
                            str(root),
                            "--registry-root",
                            str(registry_root),
                            "--output",
                            str(output),
                        ]
                    ),
                    0,
                )

            self.assertIn("rounds=1", output_text.getvalue())
            self.assertTrue(output.exists())
            self.assertTrue(output.with_name("workbench.json").exists())

            html = output.read_text(encoding="utf-8")
            self.assertIn("BadcaseFlow Workbench", html)
            self.assertIn("round-smoke", html)
            self.assertIn("workbench-agent", html)
            self.assertIn("PATCH_WORKFLOW", html)
            self.assertIn("blocked", html)

            data = json.loads(output.with_name("workbench.json").read_text(encoding="utf-8"))
            self.assertEqual(data["summary"]["rounds"], 1)
            self.assertEqual(data["summary"]["models"], 1)
            self.assertEqual(data["summary"]["actions"], 2)
            self.assertEqual(data["registry"]["models"][0]["model_id"], "workbench-agent")


if __name__ == "__main__":
    unittest.main()
