"""Save the participant's listening observations and one main tag per validation clip."""

import csv
import json
from pathlib import Path

KIT = Path(__file__).resolve().parent.parent

# Supplied by the participant after listening. These are not ASR-inferred labels.
OBSERVATIONS = {
    "val_hd_08": "number_error",
    "val_hd_01": "number_error",
    "val_hd_03": 'messing up short form "RBI" and number_error',
    "val_hd_07": 'messing up short form "GST" and number_error',
    "val_hg_01": "english_word_error",
    "val_hd_05": "speaker_mismatch; number_error",
    "val_hd_06": "speaker mismatch; number_error",
    "val_hd_02": "number_error",
    "val_hg_06": "english_word_error",
    "val_hg_05": "english_word_error",
    "val_hg_03": "english_word_error",
    "val_hd_04": 'number_error and short form "ATM" messed up',
    "val_hg_04": "english_word_error",
    "val_hg_02": "english_word_error",
}
LABELS = {
    item_id: "number_error" if item_id.startswith("val_hd_") else "english_word_error"
    for item_id in OBSERVATIONS
}


def main():
    reports = KIT / "reports" / "task1"
    with (KIT / "data" / "val_items.csv").open(encoding="utf-8", newline="") as f:
        valid_ids = {row["id"] for row in csv.DictReader(f)}
    with (reports / "listening_queue.csv").open(encoding="utf-8", newline="") as f:
        required_ids = {row["item_id"] for row in csv.DictReader(f)}
    if set(LABELS) - valid_ids or required_ids - set(LABELS):
        raise ValueError("Labels must use validation IDs and cover every required listening item.")

    out = KIT / "predictions" / "t1_val_errors.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", "category"])
        writer.writerows(LABELS.items())

    (reports / "listening_decisions.json").write_text(json.dumps(LABELS, indent=2) + "\n", encoding="utf-8")
    with (reports / "listening_notes.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", "primary_category", "observations"])
        writer.writerows((item_id, LABELS[item_id], note) for item_id, note in OBSERVATIONS.items())

    path = KIT / "code" / "task1_listen.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    matches = [cell for cell in notebook["cells"] if "LABELS = {" in "".join(cell.get("source", []))]
    if len(matches) != 1:
        raise ValueError("Expected one listening-label cell in the notebook.")
    source = "# Participant's listening decisions; full observations are in reports/task1/listening_notes.csv.\n"
    source += "# One primary category per clip; number_error is primary for the hard items.\nLABELS = {\n"
    source += "".join(f"    {item_id!r}: {tag!r},\n" for item_id, tag in LABELS.items()) + "}\n"
    matches[0]["source"] = source.splitlines(True)
    matches[0]["execution_count"] = None
    matches[0]["outputs"] = []
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    results_path = KIT / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    environment = json.loads((reports / "environment.json").read_text(encoding="utf-8"))
    if not results.get("machine") or results["machine"] == "e.g. Colab T4":
        results["machine"] = environment["platform"] + ", CPU (Task 1 scoring)"
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(LABELS)} participant-reviewed tags to {out.relative_to(KIT)}.")
    print("Preserved all secondary observations in reports/task1/listening_notes.csv.")


if __name__ == "__main__":
    main()
