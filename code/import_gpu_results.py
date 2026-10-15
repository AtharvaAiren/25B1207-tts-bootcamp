"""Import measured artifacts without overwriting local source or the submitted idea.

Usage: python code/import_gpu_results.py path/to/25B1207_tts_bootcamp.zip
The original ZIP remains the immutable copy of the remote run.
"""

import argparse
import csv
import io
import json
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

KIT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip", type=Path)
    args = parser.parse_args()
    with zipfile.ZipFile(args.zip) as source:
        names = source.namelist()
        if len(set(names)) != len(names):
            raise ValueError("Duplicate paths in ZIP")
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name:
                raise ValueError("Unsafe ZIP path")
        local = json.loads((KIT / "results.json").read_text())
        measured = json.loads(source.read("results.json"))
        if any(local[key] != measured[key] for key in ("roll_no", "seed", "task1")):
            raise ValueError("Roll, seed or Task 1 differs; inspect before importing")
        frozen_parts = lambda data: re.split(rb"(?m)^\(4\)", data, maxsplit=1)[0]
        if frozen_parts(source.read("idea.md")) != frozen_parts((KIT / "idea.md").read_bytes()):
            raise ValueError("Submitted idea card differs; never overwrite parts (1)–(3)")
        for name in ("code/normalize.py", "predictions/t2_normalized.csv",
                     "code/pronunciation.py", "code/run_postmidterm.py"):
            if name in names and (KIT / name).exists() and source.read(name) != (KIT / name).read_bytes():
                raise ValueError(f"Source differs: {name}; import matching evidence deliberately")
        staged = {"results.json": source.read("results.json")}
        # Task 4 explicitly requires both same-session timing files in code/.
        if measured.get("task4", {}).get("speedup") is not None:
            timing = {}
            for label in ("baseline", "fast"):
                name = f"code/{label}_timing.json"
                timing[label] = json.loads(source.read(name))
                staged[name] = source.read(name)
            if timing["baseline"]["session"] != timing["fast"]["session"] or any(t["n_items"] != 24 or t["seed"] != measured["seed"] for t in timing.values()):
                raise ValueError("Task 4 timing files differ in session/count/seed")
            ratio = timing["baseline"]["total_synth_s"] / timing["fast"]["total_synth_s"]
            if abs(ratio - measured["task4"]["speedup"]) > 1e-9:
                raise ValueError("Task 4 speedup differs from its timing files")
        for split, folders in (("val", ("val_control", "val_idea")),
                              ("eval", ("t2_norm", "t3_idea", "t4_fast"))):
            with (KIT / "data" / f"{split}_items.csv").open(newline="") as stream:
                ids = {row["id"] for row in csv.DictReader(stream)}
            for folder in folders:
                prefix = f"predictions/{folder}/"
                found = {name for name in names if name.startswith(prefix) and not name.endswith("/")}
                if not found:
                    continue
                if found != {prefix + item + ".wav" for item in ids}:
                    raise ValueError(f"Incomplete audio folder: {folder}")
                import soundfile as sf
                for name in found:
                    data = source.read(name)
                    info = sf.info(io.BytesIO(data))
                    if (info.format, info.channels, info.samplerate, info.subtype) != ("WAV", 1, 16000, "PCM_16") or not 0.5 <= info.duration <= 60:
                        raise ValueError(f"Invalid WAV: {name}")
                    staged[name] = data
        for name in names:
            for task in ("task2", "postmidterm"):
                prefix = f"code/{task}_evidence/"
                if name.startswith(prefix) and PurePosixPath(name).suffix in (".json", ".csv", ".log", ".md"):
                    staged[f"reports/{task}/" + name[len(prefix):]] = source.read(name)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup = KIT / "reports" / "import_backups" / stamp
        # All archive validation finishes before any existing file is replaced.
        for name, data in sorted(staged.items()):
            target = KIT / name
            if target.exists() and target.read_bytes() != data:
                saved = backup / name
                saved.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, saved)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    print(f"Imported {len(staged)} measured files; local code and idea.md preserved.")
    print("Original remote-run ZIP:", args.zip)


if __name__ == "__main__":
    main()
