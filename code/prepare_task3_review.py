"""Create a validation-only listening notebook for the measured report examples."""

import json
from pathlib import Path

KIT = Path(__file__).resolve().parent.parent


def main():
    if not (KIT / "reports/postmidterm/comparison.json").exists():
        raise RuntimeError("Import the measured post-midterm ZIP first")
    setup = '''import csv
from pathlib import Path
from IPython.display import Audio, Markdown, display

KIT = Path.cwd()
if KIT.name == "code":
    KIT = KIT.parent
EVIDENCE = KIT / "reports/postmidterm"
if not EVIDENCE.exists():
    EVIDENCE = KIT / "code/postmidterm_evidence"

def rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return {row["id"]: row for row in csv.DictReader(stream)}

ITEMS = rows(KIT / "data/val_items.csv")  # validation only
METHODS = ("baseline_provided", "control", "rewrite", "timing", "combined", "fast")
SCORES = {method: rows(EVIDENCE / f"{method}_scores.csv") for method in METHODS}

def show_item(item_id):
    item = ITEMS[item_id]
    display(Markdown(f"**{item_id} — {item['category']}**\\n\\nIntended reading: {item['spoken']}"))
    table = ["| System | CER | SIM | Failed |", "|---|---:|---:|---|"]
    for method in METHODS:
        row = SCORES[method][item_id]
        table.append(f"| {method} | {100 * float(row['cer']):.2f}% | {float(row['sim']):.4f} | {row['failed']} |")
    display(Markdown("\\n".join(table)))
    for label, folder, method in (("Provided baseline", "data/baseline_val", "baseline_provided"),
                                  ("Submitted Task 3 combination", "predictions/val_idea", "combined")):
        display(Markdown(f"**{label}**\\n\\nRecognizer transcript: {SCORES[method][item_id]['hyp']}"))
        display(Audio(filename=str(KIT / folder / f"{item_id}.wav")))
    display(Markdown("The recognizer can be wrong. Listen before making a claim about the sound. No evaluation audio is loaded."))
'''
    def markdown(text):
        return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}
    def code(text):
        return {"cell_type": "code", "metadata": {}, "source": text.splitlines(keepends=True), "execution_count": None, "outputs": []}
    cells = [markdown("# Task 3 validation review\n\nUse `.venv-task1/bin/python` as the notebook kernel. The first four examples support report Q1. The last two help investigate the timing ablation. This notebook never reads evaluation items or audio."), code(setup)]
    for item in ("val_hg_01", "val_hg_06", "val_hi_03", "val_lg_03", "val_hg_02", "val_hg_04"):
        cells.append(code(f'show_item("{item}")\n'))
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    destination = KIT / "code/task3_review.ipynb"
    if destination.exists():
        print("Preserved existing review notebook:", destination.relative_to(KIT))
        return
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Created validation-only listening notebook:", destination.relative_to(KIT))


if __name__ == "__main__":
    main()
