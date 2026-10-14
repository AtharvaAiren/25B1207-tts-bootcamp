"""Prepare Task 1 results and a notebook for human listening, after metrics.py finishes."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parent.parent
REPORTS = KIT / "reports" / "task1"
TAGS = (
    "skipped_words", "repeated_words", "number_error", "english_word_error",
    "speaker_mismatch", "other",
)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def prepare():
    summary = json.loads((REPORTS / "baseline_scores.json").read_text())
    rows = read_csv(REPORTS / "baseline_scores.csv")
    items = {r["id"]: r for r in read_csv(KIT / "data" / "val_items.csv")}
    if len(rows) != len(items) or {r["id"] for r in rows} != set(items):
        raise ValueError("The score CSV must contain all 24 validation items exactly once.")
    if summary["n_items"] != len(items):
        raise ValueError("The summary must cover all 24 validation items.")

    results_path = KIT / "results.json"
    results = json.loads((results_path if results_path.exists() else KIT / "results_template.json").read_text())
    results["task1"] = {"val_cer": summary["cer"], "val_sim": summary["sim"]}
    if not results.get("machine"):
        results["machine"] = f"{platform.system()} {platform.machine()}, CPU (Task 1 scoring)"
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    packages = {}
    for name in ("numpy", "torch", "torchaudio", "transformers", "speechbrain", "librosa", "soundfile", "huggingface-hub"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    revisions = {}
    for name in (summary["recognizer"], summary["speaker_model"]):
        ref = KIT / ".cache-task1" / "huggingface" / "hub" / ("models--" + name.replace("/", "--")) / "refs" / "main"
        revisions[name] = ref.read_text().strip() if ref.exists() else None
    scorer = KIT / "metrics.py"
    environment = {"python": sys.version, "platform": platform.platform(), "packages": packages,
                   "model_revisions": revisions, "metrics_sha256": hashlib.sha256(scorer.read_bytes()).hexdigest() if scorer.exists() else None}
    (REPORTS / "environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")

    published = json.loads((KIT / "data" / "baseline_published.json").read_text())
    for metric in ("cer", "sim"):
        delta = summary[metric] - published[metric]
        print(f"{metric.upper()}: measured={summary[metric]:.4f}, published={published[metric]:.4f}, difference={delta:+.4f}")
        if abs(delta) > 0.01:
            print("  Difference exceeds 0.01; investigate before treating the baseline as reproduced.")
    print(f"Failed items: {summary['failed']}")

    required = sorted((r for r in rows if float(r["cer"]) > 0.20), key=lambda r: -float(r["cer"]))
    queue_path = REPORTS / "listening_queue.csv"
    with queue_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["item_id", "item_category", "cer", "sim", "input_text", "spoken", "asr_hyp", "audio_file", "prompt_id"])
        writer.writeheader()
        for r in required:
            item = items[r["id"]]
            writer.writerow({"item_id": r["id"], "item_category": r["category"], "cer": r["cer"], "sim": r["sim"],
                             "input_text": item["text"], "spoken": item["spoken"], "asr_hyp": r["hyp"],
                             "audio_file": f"data/baseline_val/{r['id']}.wav", "prompt_id": item["prompt_id"]})

    notebook_path = KIT / "code" / "task1_listen.ipynb"
    if notebook_path.exists():
        print(f"Kept existing {notebook_path.name} to preserve any human annotations.")
        print(f"Updated listening queue: {len(required)} items with CER > 0.20.")
        return

    cells = [
        cell("markdown", "# Task 1: listen and label\n\nThe scores have already been computed. Listen to every item below, compare it with the intended reading, and choose its main audible error. Labels are intentionally blank. A recognizer mistake is not proof of a speech error.\n\nSelect the project's `.venv-task1` Python interpreter as the notebook kernel.\n"),
        cell("code", "from pathlib import Path\nimport csv, json\nfrom IPython.display import Audio, display\n\nKIT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'metrics.py').exists())\nwith (KIT / 'data/prompts.csv').open(encoding='utf-8', newline='') as f:\n    prompts = {r['prompt_id']: r for r in csv.DictReader(f)}\nprint('Kit:', KIT)\n"),
    ]
    for r in required:
        item = items[r["id"]]
        cells.append(cell("markdown", f"## {r['id']} — {r['category']}\n\nCER: **{float(r['cer']):.3f}** · SIM: **{float(r['sim']):.3f}**\n\nInput: {item['text']}\n\nIntended reading: {item['spoken']}\n\nRecognizer output: {r['hyp']}\n\nSanity-check problems: {r['reasons'] or 'none'}\n"))
        cells.append(cell("code", f"print('Generated baseline speech')\ndisplay(Audio(filename=str(KIT / 'data/baseline_val/{r['id']}.wav')))\nprint('Reference speaker (for checking speaker mismatch)')\ndisplay(Audio(filename=str(KIT / 'data' / prompts[{item['prompt_id']!r}]['file'])))\n"))
    tags_source = "# Fill these after listening. Add other validation IDs if you hear a clear error.\n# Allowed: " + ", ".join(TAGS) + ".\n# Use no_audio_error only if the high CER comes from recognition, not the audio.\nLABELS = {\n"
    tags_source += "".join(f"    {r['id']!r}: '',\n" for r in required) + "}\n"
    cells.append(cell("markdown", "## Your listening decisions\n\nChoose one main error per item. `no_audio_error` records that you listened but found no audible error; that item is left out of the submission tags.\n"))
    cells.append(cell("code", tags_source))
    cells.append(cell("code", "allowed = {'skipped_words', 'repeated_words', 'number_error', 'english_word_error', 'speaker_mismatch', 'other', 'no_audio_error'}\nwith (KIT / 'reports/task1/listening_queue.csv').open(encoding='utf-8', newline='') as f:\n    required_ids = {r['item_id'] for r in csv.DictReader(f)}\nwith (KIT / 'data/val_items.csv').open(encoding='utf-8', newline='') as f:\n    valid_ids = {r['id'] for r in csv.DictReader(f)}\npending = sorted(i for i in required_ids if LABELS.get(i) not in allowed)\ninvalid = {i: tag for i, tag in LABELS.items() if i not in valid_ids or tag not in allowed}\nif pending or invalid:\n    raise ValueError(f'Complete your listening decisions first. Pending: {pending}; invalid: {invalid}')\ntagged = [(i, tag) for i, tag in LABELS.items() if tag != 'no_audio_error']\nif not tagged:\n    raise ValueError('No audible errors tagged. Recheck the audio before submitting Task 1.')\nout = KIT / 'predictions/t1_val_errors.csv'\nout.parent.mkdir(parents=True, exist_ok=True)\nwith out.open('w', encoding='utf-8', newline='') as f:\n    writer = csv.writer(f)\n    writer.writerow(['item_id', 'category'])\n    writer.writerows(tagged)\n(KIT / 'reports/task1/listening_decisions.json').write_text(json.dumps(LABELS, indent=2) + '\\n', encoding='utf-8')\nprint(f'Saved {len(tagged)} human-reviewed error tags to {out}')\n"))
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}}, "nbformat": 4, "nbformat_minor": 5}
    for i, notebook_cell in enumerate(cells):
        notebook_cell["id"] = f"task1-{i}"
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {notebook_path.name}: {len(required)} recordings require listening.")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    prepare()
