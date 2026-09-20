from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from badcaseflow.cli import main


ROOT = Path(__file__).resolve().parents[1]


class CliSmokeTest(unittest.TestCase):
    def test_local_badcase_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "runs" / "demo"
            seed = ROOT / "examples" / "datasets" / "seed_tasks.jsonl"
            suite = ROOT / "examples" / "datasets" / "eval_tasks.jsonl"
            sft = run_dir / "sft.jsonl"
            preference = run_dir / "preference.jsonl"

            self.assertEqual(
                main(["ingest", "--workspace", "demo-agent", "--input", str(seed), "--run", str(run_dir)]),
                0,
            )
            self.assertEqual(
                main(["rollout", "--workspace", "demo-agent", "--run", str(run_dir), "--agent", "mock"]),
                0,
            )
            self.assertEqual(
                main(["eval", "--workspace", "demo-agent", "--run", str(run_dir), "--suite", str(suite)]),
                2,
            )
            self.assertEqual(main(["analyze-cases", "--workspace", "demo-agent", "--run", str(run_dir)]), 0)
            self.assertEqual(
                main(["export", "sft", "--workspace", "demo-agent", "--run", str(run_dir), "--output", str(sft)]),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "export",
                        "preference",
                        "--workspace",
                        "demo-agent",
                        "--run",
                        str(run_dir),
                        "--output",
                        str(preference),
                    ]
                ),
                0,
            )

            report = json.loads((run_dir / "case_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["total_failed_cases"], 1)
            self.assertEqual(report["cases"][0]["recommended_action"], "PATCH_WORKFLOW")
            self.assertTrue(sft.exists())
            self.assertTrue(preference.exists())

    def test_demo_command_writes_markdown_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "runs" / "demo"
            self.assertEqual(main(["demo", "--run", str(run_dir)]), 0)
            self.assertTrue((run_dir / "case_report.md").exists())
            report = json.loads((run_dir / "case_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["cases"][0]["recommended_action"], "PATCH_WORKFLOW")

    def test_import_external_traces_can_eval_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "runs" / "external"
            traces = ROOT / "examples" / "datasets" / "external_traces.jsonl"
            suite = ROOT / "examples" / "datasets" / "eval_tasks.jsonl"
            self.assertEqual(
                main(["import-traces", "--workspace", "demo-agent", "--input", str(traces), "--run", str(run_dir)]),
                0,
            )
            self.assertEqual(
                main(["eval", "--workspace", "demo-agent", "--run", str(run_dir), "--suite", str(suite)]),
                0,
            )
            self.assertEqual(main(["analyze-cases", "--workspace", "demo-agent", "--run", str(run_dir)]), 0)
            report = json.loads((run_dir / "case_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["total_failed_cases"], 0)

    def test_remote_plan_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "runs" / "demo"
            recipe = ROOT / "examples" / "recipes" / "sft_llamafactory.example.yaml"
            self.assertEqual(main(["demo", "--run", str(run_dir)]), 0)
            self.assertEqual(main(["init-workspace", "--workspace", "demo-agent", "--root", str(root)]), 0)
            config = root / ".badcaseflow" / "remotes.json"
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
            output = run_dir / "remote_plan.json"
            self.assertEqual(
                main(
                    [
                        "remote",
                        "plan",
                        "--target",
                        "autodl",
                        "--workspace",
                        "demo-agent",
                        "--run",
                        str(run_dir),
                        "--recipe",
                        str(recipe),
                        "--config",
                        str(config),
                        "--output",
                        str(output),
                    ]
                ),
                0,
            )
            self.assertTrue(output.exists())
            self.assertTrue(output.with_suffix(".ps1").exists())


if __name__ == "__main__":
    unittest.main()
