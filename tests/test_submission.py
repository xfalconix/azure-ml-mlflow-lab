"""Keep a failed remote job from being reported or registered as successful."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from submit_job import register_model, wait_for_job


class SubmissionTests(unittest.TestCase):
    @patch("submit_job.save_record")
    @patch("submit_job.time.sleep")
    def test_wait_observes_completion_after_queued_and_running(self, sleep, save):
        client = Mock()
        client.jobs.get.side_effect = [
            SimpleNamespace(status=status) for status in ["Queued", "Running", "Completed"]
        ]
        record = {}
        wait_for_job(client, "example-job", record)
        self.assertEqual(record["status"], "Completed")

    @patch("submit_job.save_record")
    def test_failed_or_canceled_jobs_cannot_be_registered(self, save):
        for status in ["Failed", "Canceled"]:
            with self.subTest(status=status):
                client = Mock()
                client.jobs.get.return_value = SimpleNamespace(status=status)
                with self.assertRaises(RuntimeError):
                    wait_for_job(client, "example-job", {})
                with self.assertRaises(ValueError):
                    register_model(client, "example-job")
                client.models.create_or_update.assert_not_called()


if __name__ == "__main__":
    unittest.main()
