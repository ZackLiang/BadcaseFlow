from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from badcaseflow.cli import main


ROOT = Path(__file__).resolve().parents[1]


class JobRunnerTest(unittest.TestCase):
    def test_local_job_submit_run_status_logs_and_collect(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan_path = root / "launch_train.json"
            jobs_root = root / "jobs"
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
                        "job-train",
                        "--output-root",
                        str(root / "runs"),
                        "--output",
                        str(plan_path),
                    ]
                ),
                0,
            )

            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    main(
                        [
                            "job",
                            "submit",
                            "--plan",
                            str(plan_path),
                            "--root",
                            str(jobs_root),
                            "--job-id",
                            "job-local-train",
                        ]
                    ),
                    0,
                )
            self.assertIn("status=planned", output.getvalue())

            job_dir = jobs_root / "job-local-train"
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["job", "run", "--job", str(job_dir)]), 0)
            self.assertIn("status=succeeded", output.getvalue())

            status = json.loads((job_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "succeeded")
            self.assertEqual(status["commands_completed"], 1)
            self.assertTrue((root / "runs" / "train_runs" / "job-train" / "status.json").exists())

            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["job", "status", "--job", str(job_dir)]), 0)
            self.assertIn("job_id=job-local-train", output.getvalue())

            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["job", "logs", "--job", str(job_dir), "--stream", "stdout", "--tail", "5"]), 0)
            self.assertIn("train_run=", output.getvalue())

            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["job", "collect", "--job", str(job_dir)]), 0)
            self.assertIn("collection_status=skipped", output.getvalue())

            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["job", "list", "--root", str(jobs_root)]), 0)
            self.assertIn("job-local-train", output.getvalue())

    def test_ssh_job_submit_without_execute_records_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / ".badcaseflow" / "remotes.json"
            plan_path = root / "launch_eval.json"
            jobs_root = root / "jobs"
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
                        "job-eval",
                        "--output-root",
                        str(root / "runs"),
                        "--output",
                        str(plan_path),
                    ]
                ),
                0,
            )

            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    main(
                        [
                            "job",
                            "submit",
                            "--plan",
                            str(plan_path),
                            "--root",
                            str(jobs_root),
                            "--job-id",
                            "job-ssh-eval",
                        ]
                    ),
                    0,
                )
            self.assertIn("status=planned", output.getvalue())
            job_dir = jobs_root / "job-ssh-eval"
            status = json.loads((job_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "planned")
            self.assertEqual(status["launcher"], "ssh")
            plan = json.loads((job_dir / "launch_plan.json").read_text(encoding="utf-8"))
            self.assertIn("ssh -p 22 root@example.autodl", plan["commands"]["run"][0])


if __name__ == "__main__":
    unittest.main()
