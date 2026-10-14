"""Run Task 2 on a free GPU; use --stage val, eval, or package.

No Task 3 experiment is run. Generated audio is checked in a staging folder
before it is moved into the submission's predictions folder. Model settings,
assigned prompts and seeds are unchanged from the organizer baseline.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

KIT = Path(__file__).resolve().parent.parent
EVIDENCE = KIT / "reports" / "task2"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_frozen_text():
    frozen_path = EVIDENCE / "rules_frozen.json"
    if not frozen_path.exists():
        # The ZIP/GitHub layout keeps small evidence in code/. Restore only
        # this manifest; do not regenerate or inspect evaluation text.
        submitted = KIT / "code" / "task2_evidence" / "rules_frozen.json"
        if not submitted.exists():
            raise RuntimeError("Copy reports/task2/rules_frozen.json or the packaged code/task2_evidence/ folder with your kit.")
        shutil.copy2(submitted, frozen_path)
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen["rules_sha256"] != sha256(KIT / "code" / "normalize.py"):
        raise RuntimeError("Normalization rules differ from the frozen version. Do not tune using evaluation text.")
    if frozen["normalized_csv_sha256"] != sha256(KIT / "predictions" / "t2_normalized.csv"):
        raise RuntimeError("The normalized CSV has changed; use the original frozen output.")
    if frozen["baseline_config_sha256"] != sha256(KIT / "baseline" / "config.yaml"):
        raise RuntimeError("Task 2 requires the original baseline configuration; it has changed since preparation.")
    return frozen


def check_validation_ready(frozen, seed):
    path = EVIDENCE / "comparison.json"
    if not path.exists():
        raise RuntimeError("Run --stage val first, so Task 2 is checked on validation before evaluation synthesis.")
    comparison = json.loads(path.read_text(encoding="utf-8"))
    for key in ("rules_sha256", "normalized_csv_sha256", "baseline_config_sha256"):
        if comparison.get(key) != frozen[key]:
            raise RuntimeError("Validation results belong to a different rule/text/configuration version; rerun --stage val.")
    if comparison.get("seed") != seed:
        raise RuntimeError("The seed changed since validation; rerun --stage val with the matching seed.")
    if not comparison.get("failed_items_within_limit"):
        raise RuntimeError("Validation exceeded the failed-item limit; inspect validation results before evaluation synthesis.")
    if not comparison.get("passes_other_category_guards"):
        raise RuntimeError("Task 2 exceeded a 5% CER guard; inspect validation results before evaluation synthesis.")


def run_logged(arguments, name, *, show_tail=True, allow_patch_retry=False):
    """Run a fresh process, with progress and logs but no evaluation text display."""
    log = EVIDENCE / f"{name}.log"
    command = [sys.executable, "-u", *arguments]
    for attempt in range(2 if allow_patch_retry else 1):
        print("Running:", " ".join(arguments), flush=True)
        started = time.monotonic()
        with log.open("w" if attempt == 0 else "a", encoding="utf-8") as stream:
            process = subprocess.Popen(command, cwd=KIT, stdout=stream, stderr=subprocess.STDOUT)
            last_update = started
            try:
                while process.poll() is None:
                    time.sleep(5)
                    now = time.monotonic()
                    if now - last_update >= 30:
                        print(f"  {name}: still running ({int(now - started)}s); outputs save to your kit folder.", flush=True)
                        last_update = now
            except KeyboardInterrupt:
                process.terminate()
                process.wait()
                raise
        output = log.read_text(encoding="utf-8", errors="replace")
        patch_restart = "The model's own code needed the length fix" in output or "f5_tts was imported before the length fix" in output
        if process.returncode and allow_patch_retry and attempt == 0 and patch_restart:
            print("The kit applied its one-time length patch; restarting the synthesis process.", flush=True)
            continue
        if show_tail:
            print("\n".join(output.splitlines()[-16:]), flush=True)
        if process.returncode:
            raise RuntimeError(f"{name} failed (exit {process.returncode}); details are in reports/task2/{log.name}")
        return


def check_audio(directory, split):
    import numpy as np
    import soundfile as sf
    with (KIT / "data" / f"{split}_items.csv").open(newline="", encoding="utf-8") as stream:
        expected = {row["id"] for row in csv.DictReader(stream)}
    if {path.name for path in directory.iterdir()} != {f"{item}.wav" for item in expected}:
        raise RuntimeError(f"{split}: missing or extra audio files; complete the staged run first")
    for item in sorted(expected):
        info = sf.info(directory / f"{item}.wav")
        if (info.format, info.channels, info.samplerate, info.subtype) != ("WAV", 1, 16000, "PCM_16"):
            raise RuntimeError(f"{item}: incorrect submission WAV format")
        if not 0.5 <= info.duration <= 60:
            raise RuntimeError(f"{item}: duration outside 0.5–60s")
        waveform, _ = sf.read(directory / f"{item}.wav", dtype="float32")
        if waveform.size == 0 or float(np.abs(waveform).max()) < 1e-3:
            raise RuntimeError(f"{item}: silent or empty recording")
    return len(expected)


def promote_audio(staging, final):
    final.parent.mkdir(exist_ok=True)
    if final.exists():
        # Preserve previous audio outside predictions instead of deleting it.
        backup_root = EVIDENCE / "previous_audio"
        backup_root.mkdir(exist_ok=True)
        backup = backup_root / f"{final.name}_{time.time_ns()}"
        shutil.move(str(final), str(backup))
    shutil.move(str(staging), str(final))


def synthesize_split(split, folder, seed):
    staging = EVIDENCE / "staging" / folder
    run_logged([
        "baseline/run_baseline.py", "--items", f"data/{split}_items.csv", "--out", str(staging.relative_to(KIT)),
        "--text-csv", "predictions/t2_normalized.csv", "--text-column", "normalized_text", "--seed", str(seed),
    ], f"synthesize_{folder}", show_tail=split == "val", allow_patch_retry=True)
    count = check_audio(staging, split)
    timing = staging.parent / f"{folder}_timing.json"
    shutil.copy2(timing, EVIDENCE / timing.name)
    promote_audio(staging, KIT / "predictions" / folder)
    print(f"Saved {count} checked {split} WAVs to predictions/{folder}/.", flush=True)


def require_gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU is available. In Colab choose Runtime > Change runtime type > T4 GPU; in Kaggle enable GPU.")
    packages = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions() if dist.metadata.get("Name")}
    environment = {"python": sys.version.split()[0], "gpu": torch.cuda.get_device_name(0),
                   "packages": dict(sorted(packages.items())),
                   "baseline_config_sha256": sha256(KIT / "baseline" / "config.yaml")}
    (EVIDENCE / "environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("val", "eval", "package"), required=True)
    arguments = parser.parse_args()
    os.chdir(KIT)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    if arguments.stage == "package":
        run_logged(["code/build_midterm.py"], "package")
        roll = json.loads((KIT / "results.json").read_text(encoding="utf-8"))["roll_no"]
        run_logged(["check_format.py", f"{roll}_tts_bootcamp.zip"], "check_format")
        return
    frozen = check_frozen_text()
    require_gpu()
    seed = json.loads((KIT / "results.json").read_text(encoding="utf-8"))["seed"]
    if arguments.stage == "val":
        run_logged(["metrics.py", "data/baseline_val", "--json", "reports/task2/baseline_scores.json",
                    "--csv", "reports/task2/baseline_scores.csv"], "score_baseline")
        synthesize_split("val", "val_control", seed)
        run_logged(["metrics.py", "predictions/val_control", "--json", "reports/task2/val_scores.json",
                    "--csv", "reports/task2/val_scores.csv"], "score_val_control")
        run_logged(["code/record_task2_results.py"], "record_results")
        print("Review the validation scores and clips before continuing. Task 3 remains unrun.")
    else:
        check_validation_ready(frozen, seed)
        synthesize_split("eval", "t2_norm", seed)
        print("Task 2 evaluation synthesis is complete. Evaluation listening or text-based tuning is not permitted.")


if __name__ == "__main__":
    main()
