from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from badcaseflow.cli import main
from badcaseflow.evals import list_eval_adapters
from badcaseflow.recipes import build_eval_dry_run


ROOT = Path(__file__).resolve().parents[1]


class EvalAdapterTest(unittest.TestCase):
    def test_list_eval_adapters_has_builtin_and_frameworks(self) -> None:
        rows = list_eval_adapters()
        pairs = {(row["backend"], row["stage"]) for row in rows}
        self.assertIn(("builtin", "agent_trace"), pairs)
        self.assertIn(("command", "custom"), pairs)
        self.assertIn(("promptfoo", "suite"), pairs)
        self.assertIn(("opencompass", "benchmark"), pairs)
        self.assertIn(("lighteval", "benchmark"), pairs)

    def test_eval_framework_dry_run_commands(self) -> None:
        promptfoo = build_eval_dry_run(ROOT / "examples" / "evals" / "promptfoo.example.yaml")
        opencompass = build_eval_dry_run(ROOT / "examples" / "evals" / "opencompass.example.yaml")
        lighteval = build_eval_dry_run(ROOT / "examples" / "evals" / "lighteval.example.yaml")

        self.assertEqual(promptfoo["command"][:2], ["promptfoo", "eval"])
        self.assertIn("--config", promptfoo["command"])
        self.assertEqual(opencompass["command"][0], "opencompass")
        self.assertIn("--work-dir", opencompass["command"])
        self.assertEqual(lighteval["command"][:2], ["lighteval", "accelerate"])
        self.assertIn("--tasks", lighteval["command"])

    def test_cli_evals_adapters_and_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["evals", "adapters"]), 0)
            text = output.getvalue()
            self.assertIn("builtin", text)
            self.assertIn("promptfoo", text)

            self.assertEqual(
                main(
                    [
                        "evals",
                        "dry-run",
                        "--recipe",
                        str(ROOT / "examples" / "evals" / "command_eval_smoke.example.yaml"),
                        "--output-root",
                        str(Path(tmp) / "runs"),
                    ]
                ),
                0,
            )
            manifests = list((Path(tmp) / "runs" / "dry_runs").glob("*.json"))
            self.assertEqual(len(manifests), 1)
            manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
            self.assertEqual(manifest["backend"], "command")
            self.assertEqual(manifest["stage"], "custom")

    def test_builtin_eval_run_writes_report_and_inspect(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_run = root / "runs" / "demo"
            self.assertEqual(main(["demo", "--run", str(data_run)]), 0)
            recipe = root / "builtin_eval.yaml"
            recipe.write_text(
                "\n".join(
                    [
                        "recipe_id: builtin.eval.test",
                        "backend: builtin",
                        "stage: agent_trace",
                        "data:",
                        f"  traces: {data_run.as_posix()}/traces.jsonl",
                        f"  suite: {(ROOT / 'examples' / 'datasets' / 'eval_tasks.jsonl').as_posix()}",
                        "outputs:",
                        "  output_dir: runs/eval_runs/builtin-eval-test",
                        "gate:",
                        "  min_pass_rate: 0.8",
                        "  allow_failed_cases: false",
                    ]
                ),
                encoding="utf-8",
            )
            output_root = root / "runs"
            self.assertEqual(
                main(
                    [
                        "evals",
                        "run",
                        "--workspace",
                        "demo-agent",
                        "--recipe",
                        str(recipe),
                        "--output-root",
                        str(output_root),
                        "--run-id",
                        "builtin-eval-test",
                    ]
                ),
                0,
            )
            run_dir = output_root / "eval_runs" / "builtin-eval-test"
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            report = json.loads((run_dir / "eval_report.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "succeeded")
            self.assertEqual(status["quality_status"], "blocked")
            self.assertEqual(report["failed"], 1)
            self.assertEqual(status["eval_report"]["pass_rate"], 0.6667)

            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["evals", "inspect", "--run", str(run_dir)]), 0)
            self.assertIn("quality_status=blocked", output.getvalue())

            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["evals", "list", "--root", str(output_root / "eval_runs")]), 0)
            self.assertIn("builtin-eval-test", output.getvalue())

            train_recipe = ROOT / "examples" / "recipes" / "command_smoke.example.yaml"
            self.assertEqual(
                main(
                    [
                        "train",
                        "run",
                        "--workspace",
                        "demo-agent",
                        "--recipe",
                        str(train_recipe),
                        "--output-root",
                        str(output_root),
                        "--run-id",
                        "model-train",
                    ]
                ),
                0,
            )
            registry_root = root / "registry"
            self.assertEqual(
                main(
                    [
                        "registry",
                        "register-model",
                        "--root",
                        str(registry_root),
                        "--run",
                        str(output_root / "train_runs" / "model-train"),
                        "--model-id",
                        "eval-linked-candidate",
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "registry",
                        "attach-eval-run",
                        "--root",
                        str(registry_root),
                        "--model-id",
                        "eval-linked-candidate",
                        "--eval-run",
                        str(run_dir),
                    ]
                ),
                0,
            )
            registry = json.loads((registry_root / ".badcaseflow" / "registry.json").read_text(encoding="utf-8"))
            self.assertEqual(registry["evaluations"][0]["eval_run_id"], "builtin-eval-test")
            self.assertEqual(registry["promotions"][0]["decision"], "blocked")
            artifact_sources = {artifact.get("source_type") for artifact in registry["artifacts"]}
            self.assertIn("train_run", artifact_sources)
            self.assertIn("eval_run", artifact_sources)

    def test_command_eval_harness_extracts_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "runs"
            self.assertEqual(
                main(
                    [
                        "evals",
                        "run",
                        "--recipe",
                        str(ROOT / "examples" / "evals" / "command_eval_smoke.example.yaml"),
                        "--output-root",
                        str(output_root),
                        "--run-id",
                        "command-eval-smoke",
                    ]
                ),
                0,
            )
            run_dir = output_root / "eval_runs" / "command-eval-smoke"
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            metrics = [
                json.loads(line)
                for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(status["quality_status"], "approved")
            self.assertEqual(metrics[-1]["values"]["pass_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
