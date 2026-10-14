"""Check that stale or failed validation cannot unlock evaluation synthesis."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_task2


class ValidationGateTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.evidence = Path(self.directory.name)
        self.patch = patch.object(run_task2, "EVIDENCE", self.evidence)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.frozen = {"rules_sha256": "rules", "normalized_csv_sha256": "text", "baseline_config_sha256": "config"}

    def comparison(self, **changes):
        data = dict(self.frozen, seed=0, failed_items_within_limit=True, passes_other_category_guards=True)
        data.update(changes)
        (self.evidence / "comparison.json").write_text(json.dumps(data), encoding="utf-8")

    def test_no_validation_result_is_rejected(self):
        with self.assertRaises(RuntimeError):
            run_task2.check_validation_ready(self.frozen, 0)

    def test_successful_matching_validation_is_accepted(self):
        self.comparison()
        run_task2.check_validation_ready(self.frozen, 0)

    def test_failed_validation_is_rejected(self):
        for field in ("failed_items_within_limit", "passes_other_category_guards"):
            with self.subTest(field=field):
                self.comparison(**{field: False})
                with self.assertRaises(RuntimeError):
                    run_task2.check_validation_ready(self.frozen, 0)

    def test_changed_rule_text_or_settings_are_rejected(self):
        for field in self.frozen:
            with self.subTest(field=field):
                self.comparison(**{field: "different"})
                with self.assertRaises(RuntimeError):
                    run_task2.check_validation_ready(self.frozen, 0)

    def test_changed_seed_is_rejected(self):
        self.comparison()
        with self.assertRaises(RuntimeError):
            run_task2.check_validation_ready(self.frozen, 1)


if __name__ == "__main__":
    unittest.main()
