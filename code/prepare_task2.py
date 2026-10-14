"""Freeze Task 2 rules, then apply them to all 64 items without displaying eval text."""

import csv
import hashlib
import json
import re
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KIT))

from normalize import normalize
from metrics import canon


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    predictions = KIT / "predictions"
    evidence = KIT / "reports" / "task2"
    predictions.mkdir(exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)
    val = read_csv(KIT / "data" / "val_items.csv")
    if len(val) != 24 or len({row['id'] for row in val}) != 24:
        raise ValueError("Expected the kit's 24 unique validation items")

    comparisons = []
    for row in val:
        output = normalize(row["text"])
        unchanged_latin = re.findall(r"[A-Za-z]+", row["text"])
        unchanged_latin = [word for word in unchanged_latin if not (word.isupper() and len(word) > 1)]
        if re.findall(r"[A-Za-z]+", output) != unchanged_latin:
            raise ValueError(f"Ordinary English spelling changed in {row['id']}")
        # On Hindi/long items the original text is already spoken out. Hinglish
        # spelling is intentionally left Latin in Task 2 and is not compared to
        # the fully Devanagari spoken reference.
        check_reference = row["category"] != "hinglish"
        matches = any(canon(output) == canon(ref) for ref in row["spoken"].split("||"))
        if check_reference and not matches:
            raise ValueError(f"Normalized text does not match an accepted validation reading: {row['id']}")
        comparisons.append({
            "id": row["id"], "category": row["category"], "text": row["text"],
            "normalized_text": output,
            "matches_spoken_reference": matches if check_reference else "not_checked_latin_retained",
        })

    # Write this before reading evaluation inputs. The fingerprint makes the
    # exact rule version used for blind application explicit and reproducible.
    rules_hash = digest(KIT / "code" / "normalize.py")
    frozen = {"rules_sha256": rules_hash, "baseline_config_sha256": digest(KIT / "baseline" / "config.yaml"),
              "development_split": "val only",
              "validation_items_checked": len(val), "audio_generated": False,
              "note": "Text matching is not an audio CER measurement. No evaluation text or outputs were used to develop rules."}
    (evidence / "rules_frozen.json").write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")

    # Apply fixed rules to eval programmatically; never print or inspect its text.
    evaluation = read_csv(KIT / "data" / "eval_items.csv")
    if len(evaluation) != 40 or len({row['id'] for row in evaluation}) != 40:
        raise ValueError("Expected the kit's 40 unique evaluation items")
    rows = [dict(id=row["id"], split=split, text=row["text"], normalized_text=normalize(row["text"]))
            for split, items in (("val", val), ("eval", evaluation)) for row in items]
    if digest(KIT / "code" / "normalize.py") != rules_hash:
        raise RuntimeError("Normalization rules changed during blind evaluation application")
    remaining = sum(bool(re.search(r"[0-9०-९₹%]", row["normalized_text"])) for row in rows)
    if remaining:
        raise ValueError(f"{remaining} normalized rows retain a digit, rupee or percent symbol")
    with (predictions / "t2_normalized.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "split", "text", "normalized_text"])
        writer.writeheader()
        writer.writerows(rows)
    with (evidence / "validation_text_checks.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    frozen.update(normalized_rows=len(rows), normalized_csv_sha256=digest(predictions / "t2_normalized.csv"))
    (evidence / "rules_frozen.json").write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    print("Wrote predictions/t2_normalized.csv: 24 validation + 40 evaluation rows.")
    print("All 8 hard validation texts match an accepted spoken reading; ordinary English spelling is retained.")
    print("Rules SHA256:", rules_hash)
    print("Text work complete; Task 2 audio and CER/SIM measurements remain pending.")


if __name__ == "__main__":
    main()
