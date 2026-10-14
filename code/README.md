# Task 1 reproduction

Run these commands from the kit root. The provided `metrics.py` and baseline recordings are unchanged. Task 1 downloads Whisper and ECAPA for scoring; it does not download IndicF5.

The original organizer kit is required locally. When reproducing from the GitHub checkout, put its `data/`, `metrics.py`, `check_format.py`, and `results_template.json` at the checkout root first. Keep the kit data untracked. These original kit files and downloaded model weights are not included in the submission. Copies of the completed Task 1 reports are submitted under `code/task1_evidence/`; the commands below regenerate their working copies under `reports/task1/`.

## Local setup (Python 3.12)

```bash
python3.12 -m venv .venv-task1
.venv-task1/bin/python -m pip install -r code/requirements_task1_lock.txt
```

`requirements_task1.txt` lists the scoring dependencies; `requirements_task1_lock.txt` records the complete environment used, including the notebook kernel.

## Score the supplied audio

```bash
mkdir -p reports/task1
HF_HOME="$PWD/.cache-task1/huggingface" HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false \
  .venv-task1/bin/python -u metrics.py data/baseline_val \
  --json reports/task1/baseline_scores.json \
  --csv reports/task1/baseline_scores.csv \
  > reports/task1/scoring.log 2>&1
.venv-task1/bin/python code/prepare_task1.py
```

The preparation script records **measured** Task 1 CER/SIM in `results.json`, compares them with `data/baseline_published.json`, and creates `reports/task1/listening_queue.csv` and `code/task1_listen.ipynb`. Other task results remain unchanged or null. Fill in your roll number before submitting. The script preserves an existing listening notebook so it cannot overwrite your annotations.

## Listen and label

Open `code/task1_listen.ipynb` in VS Code and select `.venv-task1/bin/python` as the kernel. Run the playback cells. Every recording with CER greater than 0.20 is included, with its intended reading, recognizer output, and reference speaker. Fill `LABELS` with your own listening decisions, then run the last cell to create `predictions/t1_val_errors.csv`. You may add other validation IDs if you hear clear errors. Tags are not inferred automatically from CER or the recognizer output.

The participant supplied listening judgments for all 14 required clips. They reported number errors on the eight hard clips and English-word errors on the six Hinglish clips, with additional abbreviation errors on RBI/GST/ATM and voice mismatch on two clips. The required CSV allows one main error per item: `number_error` is recorded as primary on the hard items, and all secondary observations are retained in `listening_notes.csv`.

To reproduce the submitted labels from these recorded participant judgments:

```bash
.venv-task1/bin/python code/save_task1_labels.py
```

This also fills the notebook's label cell and replaces the template machine description in `results.json` with the recorded macOS CPU environment. It does not run a new experiment.

## Submission check

```bash
.venv-task1/bin/python code/build_midterm.py
.venv-task1/bin/python check_format.py 25B1207_tts_bootcamp.zip
```

The build copies completed scripts, an output-cleared listening notebook, scoring/listening evidence, predictions, results, and the participant's `idea.md` once supplied. It creates `25B1207_tts_bootcamp.zip` and extracts the same contents to `submission/25B1207-tts-bootcamp/` for publishing. Notebook outputs are cleared in the submitted copy because audio playback output can embed the organizer's recordings. The local working notebook retains its playback outputs.

The submitted scoring log replaces private local folder paths with `<PROJECT>` or `<USER_HOME>`. Scores, transcripts, and listening judgments are preserved.

For a Task 1-only submission, omit unfinished tasks and use null for their measurements. Create a zip named `<rollno>_tts_bootcamp.zip` containing the submission files, then run `check_format.py` on that zip before uploading. Do not include `.venv-task1/` or downloaded model caches in the zip. Task 3 idea and report files, when attempted, must be written in your own words.

AI assistance: OpenAI Codex assisted with environment setup, reproduction commands, the listening notebook generator, recording the participant's supplied labels, and submission packaging. Error observations are the participant's listening judgments. Idea-card wording must be the participant's own.
