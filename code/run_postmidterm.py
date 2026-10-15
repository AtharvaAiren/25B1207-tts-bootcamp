"""Run the submitted Task 3 hypothesis, its ablations, and the Task 4 bonus.

Stages: prepare (CPU only), val (GPU), eval (GPU, blind), package (CPU).
Validation deliberately keeps negative results. Evaluation never selects by ear
or tunes a rule. The predeclared combined method is used even if an ablation wins.
"""

import argparse
import csv
import hashlib
import inspect
import importlib.metadata
import json
import os
import platform
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

KIT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KIT))
EVIDENCE = KIT / "reports" / "postmidterm"
from pronunciation import LATIN, rewrite, timing_speed
from experiment_support import exclusive_run, run_child
from run_task2 import check_audio, check_frozen_text

SOURCES = ("code/pronunciation.py", "code/run_postmidterm.py", "code/experiment_support.py",
           "code/normalize.py", "code/run_task2.py", "code/record_task2_results.py",
           "code/requirements_gpu_server.txt", "baseline/tts.py", "baseline/patch_length.py",
           "baseline/config.yaml", "metrics.py", "idea.md", "predictions/t2_normalized.csv")
VARIANTS = ("baseline_fresh", "control", "rewrite", "timing", "combined", "fast")
SETTINGS = {"speed_bounds": [0.75, 1.25], "task3_peak_ceiling": 0.95,
            "task4_nfe_step": 16, "chosen_task3": "combined",
            "selection": "predeclared combined method; no per-item reranking"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def fingerprints():
    result = {}
    for name in SOURCES:
        data = (KIT / name).read_bytes()
        if name == "idea.md":
            # Part (4) is necessarily written after the experiment; the submitted
            # prediction remains frozen independently of that result paragraph.
            data = re.split(rb"(?m)^\(4\)", data, maxsplit=1)[0]
        result[name] = hashlib.sha256(data).hexdigest()
    return result


def prepare():
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    check_frozen_text()
    results = read_json(KIT / "results.json")
    plan = {"schema": 1, "source_sha256": fingerprints(), "seed": results["seed"],
            "settings": SETTINGS, "idea_parts_1_to_3_frozen": True,
            "data_used_for_rules": "validation only", "variants": list(VARIANTS)}
    destination = EVIDENCE / "plan.json"
    if destination.exists() and read_json(destination) != plan:
        raise RuntimeError("An experiment plan already exists with different source/settings. Archive it before validation; never alter a plan after evaluation.")
    write_json(destination, plan)
    normalized = {row["id"]: row for row in csv_rows(KIT / "predictions/t2_normalized.csv") if row["split"] == "val"}
    preview = []
    for item in csv_rows(KIT / "data/val_items.csv"):
        text = normalized[item["id"]]["normalized_text"]
        transformed, audit = rewrite(text)
        preview.append({"id": item["id"], "normalized_text": text, "rewritten_text": transformed,
                        "words": audit})
    write_json(EVIDENCE / "validation_preview.json", preview)
    print("Prepared fixed pronunciation/timing hypothesis using validation only. No audio generated.")


def check_plan():
    plan = read_json(EVIDENCE / "plan.json")
    if plan["source_sha256"] != fingerprints() or plan["settings"] != SETTINGS:
        raise RuntimeError("Source/settings changed since preparation; evaluation cannot proceed.")
    if plan["seed"] != read_json(KIT / "results.json")["seed"]:
        raise RuntimeError("Experiment seed changed")
    check_frozen_text()
    return plan


def worker_environment():
    os.environ["HF_MODULES_CACHE"] = str(Path.home() / ".cache/huggingface/modules")
    from baseline import patch_length
    patch_length.ensure_patched()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Use the lab's CUDA environment for this stage")
    (torch.ones((4, 4), device="cuda") @ torch.ones((4, 4), device="cuda")).sum().item()
    from baseline.tts import get_engine
    engine = get_engine()
    if not engine.direct:
        raise RuntimeError("The wrapper fell back to an interface that ignores decoding overrides; do not claim timing/speed ablations.")
    params = inspect.signature(engine.U.infer_process).parameters
    if not all(name in params for name in ("speed", "nfe_step")):
        raise RuntimeError("F5 runtime does not expose the required speed/nfe_step parameters")
    write_json(EVIDENCE / "environment.json", {
        "python": sys.version, "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)), "cuda_build": torch.version.cuda,
        "baseline_config": engine.cfg,
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "torchaudio", "f5_tts", "numpy", "pyarrow", "datasets",
                      "transformers", "huggingface-hub", "speechbrain")}})
    return engine


def prompts():
    rows = csv_rows(KIT / "data/prompts.csv")
    return {row["prompt_id"]: dict(row, path=KIT / "data" / row["file"]) for row in rows}


def reference_stats(engine, prompt):
    # Use exactly the baseline's processed prompt and transcript for rate estimates.
    # This does not change the prompt passed to synthesis.
    import soundfile as sf
    path, transcript = engine.U.preprocess_ref_audio_text(str(prompt["path"]), prompt["transcript"], show_info=lambda *a, **k: None)
    return transcript, sf.info(path).duration


def model_input(variant, item, normalized, reference):
    if variant == "baseline_fresh":
        return item["text"], {}, {"rewriting": [], "timing": {"applied": False}}
    source = normalized[item["id"]]
    text, changes = rewrite(source) if variant in ("rewrite", "combined", "fast") else (source, [])
    overrides, timing = {}, {"applied": False}
    if variant in ("timing", "combined", "fast"):
        speed, timing = timing_speed(source, text, *reference)
        if timing["applied"]:
            overrides["speed"] = speed
    if variant == "fast":
        overrides["nfe_step"] = SETTINGS["task4_nfe_step"]
    return text, overrides, {"rewriting": changes, "timing": timing}


def normalize_peak(wave):
    """Permitted loudness attenuation before saving; never alter speed/pitch."""
    import numpy as np
    ceiling = SETTINGS["task3_peak_ceiling"]
    peak = float(np.max(np.abs(wave))) if len(wave) else 0
    gain = min(1.0, ceiling / peak) if peak else 1.0
    return wave * gain, gain


def synthesize(engine, items, normalized, prompt_map, variant, folder, session, *, blind=False):
    import torch
    from baseline.tts import save_wav
    seed = check_plan()["seed"]
    folder.mkdir(parents=True, exist_ok=False)
    loudness_control = folder.parent / "control_loudness" if variant == "control" and not blind else None
    if loudness_control is not None:
        loudness_control.mkdir()
    # Cache rate information outside the timed loop for every system. Prompt
    # processing by the baseline engine itself remains inside its synthesis call.
    refs = {key: reference_stats(engine, prompt_map[key]) for key in {row["prompt_id"] for row in items}}
    first = prompt_map[items[0]["prompt_id"]]
    warmup_overrides = {"nfe_step": SETTINGS["task4_nfe_step"]} if variant == "fast" else {}
    engine("नमस्ते, यह एक छोटा सा परीक्षण है।", first["path"], first["transcript"], seed=seed, **warmup_overrides)
    torch.cuda.synchronize()
    per_item, audits, total = {}, [], 0.0
    for index, item in enumerate(items, 1):
        prompt = prompt_map[item["prompt_id"]]
        torch.cuda.synchronize()
        started = time.perf_counter()
        text, overrides, audit = model_input(variant, item, normalized, refs[item["prompt_id"]])
        wave = engine(text, prompt["path"], prompt["transcript"], seed=seed, **overrides)
        gain = 1.0
        if variant in ("rewrite", "timing", "combined", "fast"):
            wave, gain = normalize_peak(wave)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        total += elapsed
        save_wav(folder / f"{item['id']}.wav", wave)
        if loudness_control is not None:
            # Diagnostic copy of the *same* synthesis, attenuated before PCM
            # clipping. No additional candidate is synthesized or selected.
            level_wave, _ = normalize_peak(wave)
            save_wav(loudness_control / f"{item['id']}.wav", level_wave)
        per_item[item["id"]] = elapsed
        if not blind:
            audits.append({"id": item["id"], "overrides": overrides, "peak_gain": gain, **audit})
        # No evaluation text, pronunciation trace, waveform or ASR output is shown.
        print(f"{variant}: {index}/{len(items)} complete ({elapsed:.2f}s)", flush=True)
    check_audio(folder, "eval" if blind else "val")
    result = {"variant": variant, "session": session, "seed": seed, "n_items": len(items),
              "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "python": platform.python_version(),
              "overrides": {} if variant in ("baseline_fresh", "control") else {"per_item": True},
              "total_synth_s": total, "per_item_s": per_item, "warmup_items_excluded": 1,
              "model_loading_excluded": True, "wav_disk_write_excluded": True,
              "settings": SETTINGS if variant not in ("baseline_fresh", "control") else {},
              "per_item_overrides": variant not in ("baseline_fresh", "control"),
              "audio_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(folder.glob("*.wav"))}}
    write_json(EVIDENCE / f"{variant}_{'eval' if blind else 'val'}_timing.json", result)
    if not blind:
        write_json(EVIDENCE / f"{variant}_audit.json", audits)
    return result


def score(folder, items, prompt_map, models, label):
    import metrics
    scored = metrics.evaluate_folder(folder, items, prompt_map, models)
    summary = metrics.summarize(scored)
    write_json(EVIDENCE / f"{label}_scores.json", summary)
    with (EVIDENCE / f"{label}_scores.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in scored for key in row)))
        writer.writeheader()
        writer.writerows(scored)
    print(f"{label}: CER {summary['cer']}; SIM {summary['sim']}; failures {summary['failed']}", flush=True)
    return summary


def quality_guards(baseline, idea):
    return {"overall_cer": idea["cer"] <= baseline["cer"] * 1.10,
            "speaker_similarity": idea["sim"] >= baseline["sim"] - 0.03,
            "failed_items": idea["failed"] / idea["n_items"] <= 0.10}


def fast_eligible(idea, fast):
    return fast["cer"] <= 1.2 * idea["cer"] and fast["failed"] / fast["n_items"] <= 0.10


def promote(folder, destination, backup):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        backup.mkdir(parents=True, exist_ok=True)
        shutil.move(str(destination), str(backup / destination.name))
    shutil.move(str(folder), str(destination))


def validation_worker():
    plan = check_plan()
    engine = worker_environment()
    session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run = EVIDENCE / "runs" / session
    run.mkdir(parents=True)
    items = csv_rows(KIT / "data/val_items.csv")
    normalized = {row["id"]: row["normalized_text"] for row in csv_rows(KIT / "predictions/t2_normalized.csv") if row["split"] == "val"}
    prompt_map = prompts()
    timings = {variant: synthesize(engine, items, normalized, prompt_map, variant, run / variant, session) for variant in VARIANTS}
    # No verifier runs during the timed synthesis loops.
    import metrics
    models = metrics.Models()
    summaries = {"baseline_provided": score(KIT / "data/baseline_val", items, prompt_map, models, "baseline_provided")}
    for variant in VARIANTS:
        summaries[variant] = score(run / variant, items, prompt_map, models, variant)
    check_audio(run / "control_loudness", "val")
    summaries["control_loudness"] = score(run / "control_loudness", items, prompt_map, models, "control_loudness")
    baseline, idea = summaries["baseline_provided"], summaries["combined"]
    fast = summaries["fast"]
    fast_quality = fast_eligible(idea, fast)
    speedup = timings["baseline_fresh"]["total_synth_s"] / timings["fast"]["total_synth_s"]
    allowed_fast = fast_quality and speedup > 1.0
    guards = quality_guards(baseline, idea)
    task2_guards = {category: summaries["control"]["cer_by_category"][category] <= baseline["cer_by_category"][category] * 1.05
                    for category in ("hindi", "hinglish", "long")}
    fresh_guards = {category: summaries["control"]["cer_by_category"][category] <= summaries["baseline_fresh"]["cer_by_category"][category] * 1.05
                    for category in ("hindi", "hinglish", "long")}
    unchanged_audio = {item["id"]: timings["baseline_fresh"]["audio_sha256"][item["id"] + ".wav"] == timings["control"]["audio_sha256"][item["id"] + ".wav"]
                       for item in items if item["text"] == normalized[item["id"]]}
    comparison = {"session": session, "source_sha256": plan["source_sha256"],
                  "seed": plan["seed"], "summaries": summaries,
                  "task2_other_category_guards_vs_provided": task2_guards,
                  "task2_other_category_guards_vs_fresh": fresh_guards,
                  "unchanged_text_same_session_audio_matches": unchanged_audio,
                  "task3_guards_vs_provided": guards,
                  "task3_guards_vs_fresh": quality_guards(summaries["baseline_fresh"], idea),
                  "task3_target_relative_reduction_vs_provided": (baseline["cer_by_category"]["hinglish"] - idea["cer_by_category"]["hinglish"]) / baseline["cer_by_category"]["hinglish"],
                  "task3_target_relative_reduction_vs_control": (summaries["control"]["cer_by_category"]["hinglish"] - idea["cer_by_category"]["hinglish"]) / summaries["control"]["cer_by_category"]["hinglish"],
                  "task4_quality_pass": fast_quality, "task4_eligible": allowed_fast,
                  "task4_speedup": speedup,
                  "negative_results_retained": True,
                  "task3_has_loudness_safety": True,
                  "loudness_note": "All idea variants share fixed peak attenuation before PCM save; the normalization-only control and raw baseline use the original save path. control_loudness scores a gain-matched copy of the same normalization-only synthesis, to separate loudness/clipping effects from pronunciation/timing.",
                  "run_directory": str(run.relative_to(KIT))}
    write_json(EVIDENCE / "comparison.json", comparison)
    # All scoring and format checks finish before submitted validation audio changes.
    promote(run / "control", KIT / "predictions/val_control", run / "previous_predictions")
    promote(run / "combined", KIT / "predictions/val_idea", run / "previous_predictions")
    # Update Task 2's checkable evidence to match the newly submitted control,
    # keeping the original remote run and its logs separately.
    task2 = KIT / "reports/task2"
    shutil.copytree(task2, run / "previous_task2_evidence", ignore=shutil.ignore_patterns("staging", "previous_audio"))
    for label, target in (("baseline_provided", "baseline"), ("control", "val")):
        for suffix in ("json", "csv"):
            shutil.copy2(EVIDENCE / f"{label}_scores.{suffix}", task2 / f"{target}_scores.{suffix}")
    shutil.copy2(EVIDENCE / "control_val_timing.json", task2 / "val_control_timing.json")
    for name in ("score_baseline.log", "synthesize_val_control.log", "score_val_control.log", "record_results.log"):
        (task2 / name).write_text("Regenerated and rescored in the post-midterm run; see code/postmidterm_evidence/val_worker.log and comparison.json. Previous logs retained in the initial fetched ZIP.\n")
    from record_task2_results import main as record_task2
    record_task2()
    results = read_json(KIT / "results.json")
    results["task2"].update(val_cer=summaries["control"]["cer"], val_sim=summaries["control"]["sim"], val_cer_hard=summaries["control"]["cer_by_category"]["hard"])
    results["task3"].update(builds_on_task2=True, settings={**SETTINGS, "normalization_only_control": True},
                            val_cer=idea["cer"], val_sim=idea["sim"],
                            val_cer_targeted=idea["cer_by_category"]["hinglish"],
                            val_control_cer=summaries["control"]["cer"])
    results["_planned_task3"]["status"] = "Measured on validation; see code/postmidterm_evidence/comparison.json"
    results["_postmidterm_run"] = {"session": session, "task3_guards": guards,
                                  "task4_quality_pass": allowed_fast, "source_sha256": plan["source_sha256"]}
    results["machine"] = f"{timings['combined']['gpu']}, CUDA (same-session post-midterm synthesis and scoring)"
    if allowed_fast:
        results["task4"].update(method_name="Same pronunciation/timing method with 16 flow steps", settings={**SETTINGS, "nfe_step": 16},
                                val_cer=fast["cer"], baseline_seconds=timings["baseline_fresh"]["total_synth_s"],
                                fast_seconds=timings["fast"]["total_synth_s"], speedup=comparison["task4_speedup"])
        shutil.copy2(EVIDENCE / "baseline_fresh_val_timing.json", KIT / "code/baseline_timing.json")
        shutil.copy2(EVIDENCE / "fast_val_timing.json", KIT / "code/fast_timing.json")
    else:
        results["task4"].update(method_name="16-step attempt did not meet quality/speed requirements; bonus not submitted", settings={"nfe_step": 16},
                                val_cer=None, baseline_seconds=None, fast_seconds=None, speedup=None)
    write_json(KIT / "results.json", results)
    write_json(EVIDENCE / "validation_complete.json", {"session": session, "source_sha256": plan["source_sha256"], "task4_eligible": allowed_fast})
    write_writing_notes(comparison)
    print("Validation complete. Inspect reports/postmidterm/comparison.json and writing_notes.md. Evaluation remains unrun.")


def write_writing_notes(comparison):
    baseline = csv_rows(EVIDENCE / "baseline_provided_scores.csv")
    combined = {row["id"]: row for row in csv_rows(EVIDENCE / "combined_scores.csv")}
    deltas = sorted((float(combined[row["id"]]["cer"]) - float(row["cer"]), row["id"]) for row in baseline)
    lines = ["# Evidence for your own report wording", "", "These are measurements and prompts, not a submission report.", "",
             "Q1: Choose two improvements and two regressions; if there are no regressions, choose the two smallest gains.", "",
             "| Item | Idea CER minus provided-baseline CER |", "|---|---:|"]
    lines += [f"| {item} | {delta:+.6f} |" for delta, item in deltas]
    lines += ["", "Q2: Compare rewrite, timing, and combined scores. Did timing help beyond rewriting? What would you change?",
              "Q3: Pick one next change and explain what measured failure motivates it.", "",
              "Also write idea.md part (4) in at most 19 words; preserve parts (1)–(3) exactly.",
              "", "The problem statement requires idea.md and report.md in your own words. Each report answer is at most 75 words.",
              "", "Uniform loudness attenuation is shared by the idea variants and fast method. Compare control_loudness with control to isolate the clipping effect; compare combined with control_loudness to assess pronunciation/timing with the same loudness treatment."]
    (EVIDENCE / "writing_notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluation_worker():
    plan = check_plan()
    completed = read_json(EVIDENCE / "validation_complete.json")
    comparison = read_json(EVIDENCE / "comparison.json")
    if completed["source_sha256"] != plan["source_sha256"] or completed["session"] != comparison["session"]:
        raise RuntimeError("Validation does not match this frozen method")
    if any(comparison["summaries"][name]["failed"] / 24 > 0.10 for name in ("control", "combined")):
        raise RuntimeError("Control or chosen idea exceeded 10% validation failures; investigate before evaluation")
    # A quality regression is retained and disclosed, not silently relabeled PASS.
    # The PS permits honest negative submissions; the official scoring guards are
    # assessed on evaluation. No method is chosen using evaluation output.
    engine = worker_environment()
    session = "eval_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run = EVIDENCE / "runs" / session
    run.mkdir(parents=True)
    items = csv_rows(KIT / "data/eval_items.csv")
    normalized = {row["id"]: row["normalized_text"] for row in csv_rows(KIT / "predictions/t2_normalized.csv") if row["split"] == "eval"}
    variants = [("control", "t2_norm"), ("combined", "t3_idea")]
    if completed["task4_eligible"]:
        variants.append(("fast", "t4_fast"))
    for variant, folder in variants:
        synthesize(engine, items, normalized, prompts(), variant, run / folder, session, blind=True)
    check_plan()
    for _, folder in variants:
        promote(run / folder, KIT / "predictions" / folder, run / "previous_predictions")
    if not completed["task4_eligible"] and (KIT / "predictions/t4_fast").exists():
        promote(KIT / "predictions/t4_fast", run / "ineligible_previous_t4", run / "previous_predictions")
    write_json(EVIDENCE / "evaluation_complete.json", {"session": session, "validation_session": completed["session"],
               "source_sha256": plan["source_sha256"], "folders": [name for _, name in variants],
               "evaluation_text_or_audio_used_for_tuning": False})
    print("Blind evaluation generation complete. No evaluation listening or tuning.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "val", "eval", "package", "all"), required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(KIT)
    if args.worker:
        {"val": validation_worker, "eval": evaluation_worker}[args.stage]()
        return
    with exclusive_run(KIT):
        EVIDENCE.mkdir(parents=True, exist_ok=True)
        stages = ("val", "eval", "package") if args.stage == "all" else (args.stage,)
        for stage in stages:
            if stage == "prepare":
                prepare()
            elif stage == "package":
                run_child(KIT, ["code/build_midterm.py"], EVIDENCE / "package.log")
                roll = read_json(KIT / "results.json")["roll_no"]
                run_child(KIT, ["check_format.py", f"{roll}_tts_bootcamp.zip"], EVIDENCE / "check_format.log")
            else:
                if stage == "val" and not (EVIDENCE / "plan.json").exists():
                    prepare()
                check_plan()
                run_child(KIT, ["code/run_postmidterm.py", "--stage", stage, "--worker"],
                          EVIDENCE / f"{stage}_worker.log", show_tail=stage == "val", patch_retry=True)


if __name__ == "__main__":
    main()
