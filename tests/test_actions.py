from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from badcaseflow.cli import main


ROOT = Path(__file__).resolve().parents[1]


class IterationActionTest(unittest.TestCase):
    def test_action_update_tracks_status_owner_and_notes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            round_dir = root / "runs" / "rounds" / "round-smoke"
            registry_root = root / "registry"
            plan = round_dir / "iteration_plan.json"
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
                        "action-train",
                        "--registry-root",
                        str(registry_root),
                        "--model-id",
                        "action-agent",
                        "--attach-round-eval",
                    ]
                ),
                0,
            )

            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["action", "list", "--plan", str(plan)]), 0)
            self.assertIn("action-001", output.getvalue())
            self.assertIn("BLOCK_PROMOTION", output.getvalue())

            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    main(
                        [
                            "action",
                            "update",
                            "--plan",
                            str(plan),
                            "--action-id",
                            "action-001",
                            "--status",
                            "accepted",
                            "--owner",
                            "zack",
                            "--note",
                            "先处理阻塞门禁的失败样本",
                        ]
                    ),
                    0,
                )
            self.assertIn("status=accepted", output.getvalue())
            self.assertIn("owner=zack", output.getvalue())

            updated = json.loads(plan.read_text(encoding="utf-8"))
            action = updated["actions"][0]
            self.assertEqual(action["status"], "accepted")
            self.assertEqual(action["owner"], "zack")
            self.assertEqual(action["notes"][0]["text"], "先处理阻塞门禁的失败样本")
            self.assertEqual(updated["summary"]["action_status_counts"]["accepted"], 1)
            self.assertTrue(action["history"])

            markdown = plan.with_suffix(".md").read_text(encoding="utf-8")
            self.assertIn("| action-001 | accepted | zack |", markdown)

            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["action", "list", "--plan", str(plan), "--status", "accepted"]), 0)
            self.assertIn("action-001", output.getvalue())
            self.assertNotIn("action-002", output.getvalue())


if __name__ == "__main__":
    unittest.main()
