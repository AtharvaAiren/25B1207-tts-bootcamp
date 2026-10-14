"""Record Task 2 measurements only after all 24 validation WAVs have been scored."""

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KIT))

from metrics import summarize


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def scored_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        if row["failed"] not in ("True", "False"):
            raise ValueError(f"Invalid failed flag in {path.name}")
        row["failed"] = row["failed"] == "True"
        row["cer"], row["sim"] = float(row["cer"]), float(row["sim"])
        if not math.isfinite(row["cer"]) or not math.isfinite(row["sim"]):
            raise ValueError("Non-finite measured score")
    return rows


def main():
    evidence = KIT / "reports" / "task2"
    with (KIT / "data" / "val_items.csv").open(newline="", encoding="utf-8") as stream:
        expected = {row["id"]: row["category"] for row in csv.DictReader(stream)}
    baseline = read_json(evidence / "baseline_scores.json")
    measured = read_json(evidence / "val_scores.json")
    for prefix, summary in (("baseline", baseline), ("val", measured)):
        rows = scored_rows(evidence / f"{prefix}_scores.csv")
        if len(rows) != 24 or {row['id']: row['category'] for row in rows} != expected:
            raise ValueError(f"{prefix}: scores must contain all 24 unique validation items")
        recomputed = summarize(rows)
        if any(summary.get(key) != value for key, value in recomputed.items()):
            raise ValueError(f"{prefix}: JSON summary differs from its per-item measured CSV")
    if baseline["failed"]:
        raise ValueError("Provided baseline recordings unexpectedly failed; inspect the scoring environment")
    audio = KIT / "predictions" / "val_control"
    if {path.name for path in audio.iterdir()} != {f"{item}.wav" for item in expected}:
        raise ValueError("Need exactly 24 generated validation-control WAV files")

    results = read_json(KIT / "results.json")
    timing = read_json(evidence / "val_control_timing.json")
    if timing["n_items"] != 24 or timing["seed"] != results["seed"] or timing["overrides"]:
        raise ValueError("Task 2 requires matching seed and unchanged baseline decoding settings")
    frozen = read_json(evidence / "rules_frozen.json")
    rules_hash = hashlib.sha256((KIT / "code" / "normalize.py").read_bytes()).hexdigest()
    if frozen["rules_sha256"] != rules_hash:
        raise ValueError("Normalization code changed after its frozen application")
    if frozen["baseline_config_sha256"] != hashlib.sha256((KIT / "baseline" / "config.yaml").read_bytes()).hexdigest():
        raise ValueError("Baseline configuration changed during the Task 2 run")

    guards = {
        category: {
            "baseline_cer": baseline["cer_by_category"][category],
            "normalized_cer": measured["cer_by_category"][category],
            "limit": round(baseline["cer_by_category"][category] * 1.05, 6),
            "passes": measured["cer_by_category"][category] <= baseline["cer_by_category"][category] * 1.05,
        } for category in ("hindi", "hinglish", "long")
    }
    base_hard = baseline["cer_by_category"]["hard"]
    report = {
        "rules_sha256": rules_hash,
        "normalized_csv_sha256": frozen["normalized_csv_sha256"],
        "baseline_config_sha256": frozen["baseline_config_sha256"],
        "seed": timing["seed"],
        "baseline": baseline,
        "normalization_only": measured,
        "hard_relative_cer_reduction": (base_hard - measured["cer_by_category"]["hard"]) / base_hard if base_hard else None,
        "other_category_guards": guards,
        "passes_other_category_guards": all(guard["passes"] for guard in guards.values()),
        "failed_items_within_limit": measured["failed"] / 24 <= 0.10,
        "note": "Actual audio scores. Reference-text matching in validation_text_checks.csv is a separate text-only check.",
    }
    (evidence / "comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    results["task2"].update(val_cer=measured["cer"], val_sim=measured["sim"],
                            val_cer_hard=measured["cer_by_category"]["hard"])
    results.setdefault("_task1_machine", results["machine"])
    results["machine"] = f"{timing['gpu']}, Python {timing['python']} (Task 2 synthesis and scoring)"
    results["_task2_run"] = {
        "rules_sha256": rules_hash, "seed": timing["seed"], "baseline_decoding_unchanged": True,
        "baseline_score_source": "reports/task2/baseline_scores.json (provided audio rescored in the same environment)",
        "passes_other_category_guards": report["passes_other_category_guards"],
        "failed_items": measured["failed"],
    }
    (KIT / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Recorded Task 2: CER {measured['cer']}, SIM {measured['sim']}, hard CER {measured['cer_by_category']['hard']}.")
    print("Other-category guards:", "PASS" if report["passes_other_category_guards"] else "FAIL — inspect validation results")
    if not report["failed_items_within_limit"]:
        raise SystemExit("More than 10% failed validation audio: this run is not eligible for scoring.")


if __name__ == "__main__":
    main()
