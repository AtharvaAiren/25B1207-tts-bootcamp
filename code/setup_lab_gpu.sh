#!/usr/bin/env bash
# Install into this project's own environment, for the participant's Blackwell GPU.
set -euo pipefail

task_kit_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_kit_dir"
task_python="${TTS_PYTHON:-python3}"
"$task_python" -c 'import sys; assert (3, 10) <= sys.version_info[:2] < (3, 13), "Use Python 3.10–3.12; the lab has Python 3.12.3."'

if [ ! -d .venv-tts ]
then
    "$task_python" -m venv .venv-tts
fi
source .venv-tts/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r code/requirements_gpu_server.txt
python -m pip check
python code/server_preflight.py --check-runtime

printf '%s\n' 'Setup checks passed. In your terminal run: source .venv-tts/bin/activate'
printf '%s\n' 'Then log in to Hugging Face and run: python code/run_task2.py --stage val'
