"""Lab diagnostics and import checks. No model weights or credentials printed.

--check-runtime applies the organizer's provided F5 length patch before imports.
"""

import argparse
import importlib.metadata
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KIT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-runtime", action="store_true", help="require GPU calculation and synthesis/scoring imports")
    options = parser.parse_args()
    print("Python:", sys.version.split()[0])
    print("Platform:", platform.system(), platform.machine())
    if not (3, 10) <= sys.version_info[:2] < (3, 13):
        print("Use Python 3.10–3.12 for the repaired dependency list; the lab has Python 3.12.3.")
    # The organizer's patcher searches this location. A custom weight cache is
    # fine, but downloaded Python modules must remain where its patch can find them.
    os.environ["HF_MODULES_CACHE"] = str(Path.home() / ".cache" / "huggingface" / "modules")
    executable = shutil.which("nvidia-smi")
    if executable:
        try:
            result = subprocess.run(
                [executable, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            )
            print("NVIDIA GPU / driver / memory:")
            if result.returncode == 0:
                print(result.stdout.strip())
            else:
                print("nvidia-smi could not query a GPU; check your lab's GPU allocation.")
        except subprocess.TimeoutExpired:
            print("nvidia-smi timed out; check your GPU allocation.")
    else:
        print("nvidia-smi: not available in this environment")

    gpu_ready = False
    try:
        import torch
        print("PyTorch:", torch.__version__)
        print("PyTorch CUDA build:", torch.version.cuda)
        available = torch.cuda.is_available()
        print("CUDA available to Python:", available)
        if available:
            print("Visible GPU:", torch.cuda.get_device_name(0))
            matrix = torch.ones((4, 4), device="cuda")
            (matrix @ matrix).sum().item()
            gpu_ready = True
            print("CUDA calculation: passed")
        else:
            print("Use an allocated NVIDIA GPU and a compatible CUDA-enabled PyTorch installation.")
    except ImportError:
        print("PyTorch: not installed in this Python environment")
    except (RuntimeError, OSError) as error:
        print("PyTorch could not load:", type(error).__name__)

    print("f5_tts installed:", importlib.util.find_spec("f5_tts") is not None)
    for name in ("transformers", "huggingface-hub", "torchaudio", "speechbrain", "numpy"):
        try:
            print(f"{name}: {importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            print(f"{name}: not installed")
    if options.check_runtime:
        if not gpu_ready:
            raise SystemExit("GPU runtime check failed; resolve the CUDA environment before synthesis.")
        # Use the organizer's supplied patch before importing its F5 dependency.
        # No model, ASR or vocoder weights are loaded by these import checks.
        from baseline.patch_length import ensure_patched
        ensure_patched()
        from f5_tts.infer import utils_infer
        from transformers import AutoModel
        from speechbrain.inference.speaker import EncoderClassifier
        print("F5-TTS, Transformers and SpeechBrain imports: passed")
    print("No synthesis or evaluation-text inspection was performed.")


if __name__ == "__main__":
    main()
