"""Build a source-only update for the existing lab folder; no results or kit data."""

import json
import zipfile
from pathlib import Path

KIT = Path(__file__).resolve().parent.parent


def main():
    roll = json.loads((KIT / "results.json").read_text())["roll_no"]
    output = KIT / f"{roll}_tts_postmidterm_update.zip"
    files = [path for path in (KIT / "code").iterdir() if path.is_file() and path.suffix in {".py", ".txt", ".md", ".ipynb"}]
    # The runner freezes its own plan on first start. Do not copy local plans or
    # old results over the lab's measured artifacts.
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as target:
        for path in sorted(files):
            data = path.read_bytes()
            if path.suffix == ".ipynb":
                notebook = json.loads(data)
                for cell in notebook["cells"]:
                    if cell["cell_type"] == "code":
                        cell["outputs"] = []
                        cell["execution_count"] = None
                notebook.get("metadata", {}).pop("widgets", None)
                data = (json.dumps(notebook, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
            target.writestr(path.relative_to(KIT).as_posix(), data)
    print(f"Source-only lab update: {output.name} ({len(files)} code files)")
    print("No results.json, idea.md, predictions, kit data or model weights are replaced.")


if __name__ == "__main__":
    main()
