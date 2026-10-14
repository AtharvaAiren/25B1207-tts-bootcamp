"""Prepare a local-only GPU input ZIP: kit + participant work, no weights/caches.

This is a convenience transfer file for your own Colab/Kaggle runtime. It is
not the submission ZIP and must not be pushed to the public repository,
because it contains the organizer's data folder.
"""

import json
import zipfile
from pathlib import Path

KIT = Path(__file__).resolve().parent.parent


def main():
    roll = json.loads((KIT / "results.json").read_text(encoding="utf-8"))["roll_no"]
    submission = KIT / f"{roll}_tts_bootcamp.zip"
    if not submission.exists():
        raise RuntimeError("Run code/build_midterm.py before preparing the GPU input bundle")
    with zipfile.ZipFile(submission) as source:
        files = {name: source.read(name) for name in source.namelist() if not name.endswith("/")}
    sources = [KIT / name for name in ("metrics.py", "check_format.py", "results_template.json")]
    for folder in ("baseline", "data"):
        sources.extend(path for path in (KIT / folder).rglob("*") if path.is_file())
    for source in sources:
        relative = source.relative_to(KIT)
        if source.is_symlink() or any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
            continue
        if source.suffix not in {".py", ".yaml", ".txt", ".ipynb", ".csv", ".json", ".wav"}:
            continue
        data = source.read_bytes()
        if source.suffix == ".ipynb":
            notebook = json.loads(data)
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    cell["outputs"] = []
                    cell["execution_count"] = None
            data = (json.dumps(notebook, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        files[relative.as_posix()] = data
    output = KIT / f"{roll}_tts_gpu_input.zip"
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as target:
        for name, data in sorted(files.items()):
            target.writestr(name, data)
    print(f"GPU input bundle: {output.name} ({output.stat().st_size / 1e6:.2f} MB; {len(files)} files)")
    print("Upload this to your own Drive/Kaggle workspace. It includes kit data and is excluded from public GitHub.")


if __name__ == "__main__":
    main()
