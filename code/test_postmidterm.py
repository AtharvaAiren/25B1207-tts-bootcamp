"""CPU tests for the experiment's scientific controls and audio safety."""

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import pronunciation as p
import run_postmidterm as experiment
from experiment_support import exclusive_run


class PronunciationTests(unittest.TestCase):
    def test_submitted_syllable_examples(self):
        for word, expected in (("internet", 3), ("slow", 1), ("movie", 2), ("shopping", 2)):
            with self.subTest(word=word):
                self.assertEqual(p.syllables(word), expected)

    def test_patterns_compose_on_unseen_combinations(self):
        # Neither word occurs in the validation items: shared spelling patterns
        # must work without a full-word lookup or an ID-based replacement.
        for word, expected in (("sleeping", "स्लीपिंग"), ("shipping", "शिपिंग")):
            self.assertEqual(p.rewrite(word)[0], expected)

    def test_unsupported_words_and_hindi_are_preserved(self):
        text = "मेरा Qxough can't पढ़ना ठीक है।"
        self.assertEqual(p.rewrite(text)[0], text)
        self.assertEqual(p.rewrite("नमस्ते दुनिया।")[0], "नमस्ते दुनिया।")

    def test_speed_is_bounded_and_no_english_is_unchanged(self):
        self.assertEqual(p.timing_speed("नमस्ते", "नमस्ते", "मेरा नाम राम है", 8)[0], 1)
        for reference in ("मैं", "यह एक अपेक्षाकृत लंबा वाक्य है।"):
            for target in ("slow", "internet", "shopping " * 20):
                speed, audit = p.timing_speed(target, p.rewrite(target)[0], reference, 10)
                self.assertGreaterEqual(speed, 0.75)
                self.assertLessEqual(speed, 1.25)
                self.assertTrue(audit["applied"])


class ExperimentControlsTests(unittest.TestCase):
    def test_ablations_change_only_their_declared_components(self):
        item = {"id": "synthetic", "text": "आज office में 2 हैं।"}
        normalized = {"synthetic": "आज office में दो हैं।"}
        reference = ("मैं आपका स्वागत करता हूँ।", 8)
        raw, raw_settings, _ = experiment.model_input("baseline_fresh", item, normalized, reference)
        control, settings, _ = experiment.model_input("control", item, normalized, reference)
        rewrite, rewrite_settings, _ = experiment.model_input("rewrite", item, normalized, reference)
        timing, timing_settings, _ = experiment.model_input("timing", item, normalized, reference)
        combined, combined_settings, _ = experiment.model_input("combined", item, normalized, reference)
        fast, fast_settings, _ = experiment.model_input("fast", item, normalized, reference)
        self.assertEqual(raw, item["text"])
        self.assertEqual(raw_settings, {})
        self.assertEqual(control, normalized["synthetic"])
        self.assertEqual(settings, {})
        self.assertNotEqual(rewrite, control)
        self.assertEqual(rewrite_settings, {})
        self.assertEqual(timing, control)
        self.assertIn("speed", timing_settings)
        self.assertEqual(combined, rewrite)
        self.assertEqual(fast, combined)
        self.assertEqual(fast_settings["speed"], combined_settings["speed"])
        self.assertEqual(fast_settings["nfe_step"], 16)

    def test_loudness_is_only_uniform_attenuation(self):
        wave = np.array([-2, -1, 0, 1, 2], dtype="float32")
        result, gain = experiment.normalize_peak(wave)
        np.testing.assert_array_equal(result, wave * gain)
        self.assertLessEqual(float(np.abs(result).max()), 0.950001)
        safe = wave / 4
        np.testing.assert_array_equal(experiment.normalize_peak(safe)[0], safe)

    def test_bonus_requires_quality_and_failure_limit(self):
        idea = {"cer": 0.1}
        self.assertTrue(experiment.fast_eligible(idea, {"cer": 0.12, "failed": 0, "n_items": 24}))
        self.assertFalse(experiment.fast_eligible(idea, {"cer": 0.121, "failed": 0, "n_items": 24}))
        self.assertFalse(experiment.fast_eligible(idea, {"cer": 0.1, "failed": 3, "n_items": 24}))

    def test_second_runner_is_rejected_and_lock_releases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with exclusive_run(root):
                with self.assertRaises(RuntimeError):
                    with exclusive_run(root):
                        self.fail("Concurrent runner acquired the lock")
            with exclusive_run(root):
                pass

    def test_full_artifact_pipeline_with_synthetic_audio_and_fake_models(self):
        """Exercise staging, recording, bonus gating and blind output contracts.

        All inputs here are synthetic; no real model inference or quality claim.
        """
        import soundfile as sf
        import metrics
        import run_task2
        import record_task2_results
        from contextlib import ExitStack
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as directory, ExitStack() as patches:
            root = Path(directory)
            evidence = root / "reports/postmidterm"
            task2 = root / "reports/task2"
            task2.mkdir(parents=True)
            (root / "data/prompts").mkdir(parents=True)
            (root / "data/baseline_val").mkdir()
            (root / "predictions").mkdir()
            (root / "code").mkdir()
            (root / "baseline").mkdir()
            wave = np.sin(np.arange(16000) * 0.08).astype("float32") * 0.1
            sf.write(root / "data/prompts/V1.wav", wave, 16000, subtype="PCM_16")
            prompt = "prompt_id,file,transcript\nV1,prompts/V1.wav,मेरा नाम राम है।\n"
            (root / "data/prompts.csv").write_text(prompt)
            categories = ["hindi"] * 6 + ["hinglish"] * 6 + ["hard"] * 8 + ["long"] * 4
            all_rows = []
            for split, count in (("val", 24), ("eval", 40)):
                rows = []
                for index in range(count):
                    item = f"{split}_synthetic_{index:02d}"
                    text = "यह office है।"
                    row = {"id": item, "category": categories[index] if split == "val" else "hinglish",
                           "text": text, "prompt_id": "V1"}
                    if split == "val":
                        row["spoken"] = "यह ऑफिस है।"
                        sf.write(root / f"data/baseline_val/{item}.wav", wave, 16000, subtype="PCM_16")
                    rows.append(row)
                    all_rows.append({"id": item, "split": split, "text": text, "normalized_text": text})
                with (root / f"data/{split}_items.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            with (root / "predictions/t2_normalized.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
                writer.writeheader()
                writer.writerows(all_rows)
            (root / "code/normalize.py").write_text("# synthetic fixed rules\n")
            (root / "baseline/config.yaml").write_text("synthetic: true\n")
            fingerprint = lambda name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            experiment.write_json(task2 / "rules_frozen.json", {
                "rules_sha256": fingerprint("code/normalize.py"),
                "normalized_csv_sha256": fingerprint("predictions/t2_normalized.csv"),
                "baseline_config_sha256": fingerprint("baseline/config.yaml")})
            template = {"seed": 0, "machine": "synthetic", "task2": {}, "task3": {}, "task4": {}, "_planned_task3": {}}
            experiment.write_json(root / "results.json", template)
            prep = lambda *args, **kwargs: (str(root / "data/prompts/V1.wav"), "मेरा नाम राम है।")

            class FakeEngine:
                U = SimpleNamespace(preprocess_ref_audio_text=staticmethod(prep))
                def __call__(self, *args, **kwargs):
                    return wave.copy()

            def fake_scores(folder, items, prompts, models):
                value = {"baseline_val": 0.2, "baseline_fresh": 0.2, "control": 0.15,
                         "combined": 0.1, "fast": 0.11}.get(folder.name, 0.13)
                return [{"id": row["id"], "category": row["category"], "cer": value,
                         "sim": 0.85, "failed": False, "hyp": "synthetic", "reasons": ""} for row in items]

            for module in (experiment, run_task2, record_task2_results):
                patches.enter_context(patch.object(module, "KIT", root))
            patches.enter_context(patch.object(experiment, "EVIDENCE", evidence))
            patches.enter_context(patch.object(run_task2, "EVIDENCE", task2))
            patches.enter_context(patch.object(experiment, "fingerprints", return_value={"synthetic": "fixed"}))
            patches.enter_context(patch.object(experiment, "worker_environment", return_value=FakeEngine()))
            patches.enter_context(patch("torch.cuda.synchronize"))
            patches.enter_context(patch("torch.cuda.get_device_name", return_value="Synthetic GPU"))
            patches.enter_context(patch.object(metrics, "Models", return_value=object()))
            patches.enter_context(patch.object(metrics, "evaluate_folder", side_effect=fake_scores))
            patches.enter_context(patch("builtins.print"))
            experiment.prepare()
            experiment.validation_worker()
            measured = experiment.read_json(root / "results.json")
            self.assertEqual(measured["task3"]["val_cer"], 0.1)
            self.assertEqual(measured["task2"]["val_cer"], 0.15)
            self.assertEqual(len(list((root / "predictions/val_idea").glob("*.wav"))), 24)
            self.assertEqual(len(list((root / "predictions/val_control").glob("*.wav"))), 24)
            self.assertIn("control_loudness", experiment.read_json(evidence / "comparison.json")["summaries"])
            experiment.evaluation_worker()
            self.assertEqual(len(list((root / "predictions/t2_norm").glob("*.wav"))), 40)
            self.assertEqual(len(list((root / "predictions/t3_idea").glob("*.wav"))), 40)
            complete = experiment.read_json(evidence / "evaluation_complete.json")
            self.assertFalse(complete["evaluation_text_or_audio_used_for_tuning"])
            eligible = experiment.read_json(evidence / "validation_complete.json")["task4_eligible"]
            self.assertEqual((root / "predictions/t4_fast").exists(), eligible)


if __name__ == "__main__":
    unittest.main()
