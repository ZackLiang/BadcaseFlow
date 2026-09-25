from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from badcaseflow import __version__
from badcaseflow.cli import main


ROOT = Path(__file__).resolve().parents[1]


class CliSmokeTest(unittest.TestCase):
    def test_version_command(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["version"])
        self.assertEqual(code, 0)
        self.assertIn(f"badcaseflow={__version__}", output.getvalue())

        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["version", "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["name"], "badcaseflow")
        self.assertEqual(payload["version"], __version__)

    def test_doctor_reports_repo_readiness(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["doctor", "--root", str(ROOT)])
        self.assertEqual(code, 0)
        text = output.getvalue()
        self.assertIn("status=ok", text)
        self.assertIn("training_adapters=", text)
        self.assertIn("eval_adapters=", text)
        self.assertIn("README.md=ok", text)
        self.assertIn("CHANGELOG.md=ok", text)

        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["doctor", "--root", str(ROOT), "--json"])
        self.assertEqual(code, 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["badcaseflow_version"], __version__)
        self.assertGreaterEqual(report["training_adapters"], 2)
        self.assertGreaterEqual(report["eval_adapters"], 2)

    def test_doctor_strict_fails_when_release_files_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(["doctor", "--root", tmp, "--json"])
            self.assertEqual(code, 0)
            report = json.loads(output.getvalue())
            self.assertEqual(report["status"], "warning")
            self.assertIn("README.md", report["missing"])

            output = io.StringIO()
            with redirect_stdout(output):
                code = main(["doctor", "--root", tmp, "--strict"])
            self.assertEqual(code, 2)

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
