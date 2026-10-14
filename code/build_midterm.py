"""Build the current submission ZIP and extract it to an isolated GitHub checkout."""

import json
import zipfile
from pathlib import Path

KIT = Path(__file__).resolve().parent.parent
SKIP_PARTS = {"__pycache__", ".git", ".venv-task1", ".cache-task1"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".pt", ".pth", ".ckpt", ".safetensors", ".bin"}


def add_tree(files, source, target):
    if not source.exists():
        return
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if not path.is_file() or path.is_symlink() or any(part in SKIP_PARTS for part in relative.parts):
            continue
        if path.name == ".DS_Store" or path.suffix.lower() in SKIP_SUFFIXES:
            continue
        data = path.read_bytes()
        if path.suffix == ".log":
            # Public logs retain measurements while removing private local paths.
            text = data.decode("utf-8")
            text = text.replace(str(KIT.resolve()), "<PROJECT>")
            text = text.replace(str(Path.home()), "<USER_HOME>")
            data = text.encode("utf-8")
        if path.suffix == ".ipynb":
            notebook = json.loads(data)
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    cell["outputs"] = []
                    cell["execution_count"] = None
            # Audio playback outputs can contain copies of the kit recordings.
            notebook.get("metadata", {}).pop("widgets", None)
            data = (json.dumps(notebook, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        files[(target / relative).as_posix()] = data


def add_task2_evidence(files):
    # Include small, checkable measurements and the rule fingerprint. Staged or
    # backed-up generated audio must not be duplicated inside code/.
    evidence = KIT / "reports" / "task2"
    if not evidence.exists():
        return
    allowed = {
        "rules_frozen.json", "validation_text_checks.csv", "environment.json",
        "baseline_scores.json", "baseline_scores.csv", "val_scores.json", "val_scores.csv",
        "comparison.json", "val_control_timing.json", "t2_norm_timing.json",
        "score_baseline.log", "synthesize_val_control.log", "score_val_control.log",
        "record_results.log", "synthesize_t2_norm.log",
    }
    for name in sorted(allowed):
        path = evidence / name
        if not path.is_file() or path.is_symlink():
            continue
        data = path.read_bytes()
        if path.suffix == ".log":
            text = data.decode("utf-8")
            text = text.replace(str(KIT.resolve()), "<PROJECT>").replace(str(Path.home()), "<USER_HOME>")
            data = text.encode("utf-8")
        files[f"code/task2_evidence/{name}"] = data


def main():
    results = json.loads((KIT / "results.json").read_text(encoding="utf-8"))
    roll_no = results["roll_no"]
    if not roll_no or not roll_no.isascii() or not roll_no.isalnum():
        raise ValueError("results.json must contain an alphanumeric roll number.")
    files = {}
    add_tree(files, KIT / "code", Path("code"))
    add_tree(files, KIT / "reports" / "task1", Path("code") / "task1_evidence")
    add_task2_evidence(files)
    add_tree(files, KIT / "predictions", Path("predictions"))
    for name in ("results.json", "idea.md", "report.md"):
        source = KIT / name
        if source.exists():
            files[name] = source.read_bytes()

    files[".gitignore"] = (
        "/data/\n/.cache-task1/\n/.venv-task1/\n/.venv-tts/\n/submission/\n/reports/\n"
        "__pycache__/\n*.pyc\n.DS_Store\n*.zip\n*.pt\n*.pth\n*.ckpt\n*.safetensors\n*.bin\n"
    ).encode("utf-8")
    status = "Included in idea.md." if "idea.md" in files else "Pending the participant's parts (1)–(3)."
    task2_text = (KIT / "predictions" / "t2_normalized.csv").exists()
    task2_audio = (KIT / "predictions" / "t2_norm").is_dir()
    if task2_audio:
        task2_status = "Task 2: normalized text and generated evaluation audio are included. "
    elif task2_text:
        task2_status = "Task 2: all 64 normalized texts are included; evaluation audio is pending. "
    else:
        task2_status = "Task 2 is not completed; no generated evaluation audio is submitted yet. "
    if task2_text:
        task2 = results.get("task2", {})
        if task2.get("val_cer") is None:
            task2_status += "Validation audio scores remain pending. "
        else:
            task2_status += f"Measured validation CER {task2['val_cer']}, SIM {task2['val_sim']}. "
    files["README.md"] = (
        f"# {roll_no} TTS bootcamp submission\n\n"
        "Task 1: 24 supplied baseline recordings scored; CER 0.2426, SIM 0.8303, zero failed items. "
        "The participant reviewed all 14 recordings above CER 0.20; their tags are in "
        "[predictions/t1_val_errors.csv](predictions/t1_val_errors.csv).\n\n"
        "Reproduction commands and AI-assistance disclosure: [code/README.md](code/README.md). "
        "Scoring and listening evidence: [code/task1_evidence/](code/task1_evidence/).\n\n"
        f"Idea card: {status}\n\n"
        + task2_status +
        "The original organizer kit is a local prerequisite for reproduction; its data and model weights "
        "are excluded from this repository.\n"
    ).encode("utf-8")
    overview = KIT / "code" / "repository_overview.md"
    if overview.exists():
        # Preserve the participant's repository overview rather than overwriting
        # it during packaging. Refresh its Task 2 status after actual measurement.
        text = overview.read_text(encoding="utf-8")
        pending = "Task 2 evaluation audio and validation audio scores remain pending."
        task2 = results.get("task2", {})
        if task2.get("val_cer") is not None:
            state = "Task 2 evaluation audio is included." if task2_audio else "Task 2 evaluation audio remains pending."
            state += f" Measured validation CER is {task2['val_cer']} and SIM is {task2['val_sim']}."
            text = text.replace(pending, state)
        files["README.md"] = text.encode("utf-8")

    for name, data in files.items():
        if str(KIT.resolve()).encode() in data or str(Path.home()).encode() in data:
            raise ValueError(f"Remove private local paths before publishing: {name}")

    archive = KIT / f"{roll_no}_tts_bootcamp.zip"
    previous = {}
    if archive.exists():
        with zipfile.ZipFile(archive) as zipped:
            previous = {name: zipped.read(name) for name in zipped.namelist() if not name.endswith("/")}
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for name, data in sorted(files.items()):
            zipped.writestr(name, data)
    if archive.stat().st_size > 100_000_000:
        raise ValueError("Submission exceeds 100 MB.")

    checkout = KIT / "submission" / f"{roll_no}-tts-bootcamp"
    checkout.mkdir(parents=True, exist_ok=True)
    for name in previous.keys() - files.keys():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or ".git" in relative.parts:
            raise ValueError(f"Unsafe prior archive path: {name}")
        old = checkout / relative
        if old.exists():
            if not old.is_file() or old.is_symlink() or old.read_bytes() != previous[name]:
                raise ValueError(f"Preserve modified checkout file before rebuilding: {name}")
            old.unlink()
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(checkout)
    print(f"ZIP: {archive.name} ({archive.stat().st_size / 1e6:.3f} MB; {len(files)} files)")
    print(f"Extracted checkout: {checkout.relative_to(KIT)}")
    print("Idea card: " + ("present" if "idea.md" in files else "MISSING — still required for the midterm form"))


if __name__ == "__main__":
    main()
