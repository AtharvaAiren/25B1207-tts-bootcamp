# Reproduction and measured status

The fetched first Task 2 validation run is now included. Same-environment rescoring of the supplied baseline gives CER 0.2421 and SIM 0.8303; normalization gives CER 0.1616 and SIM 0.8078, with one clipping failure. Hard-category CER falls from 0.4583 to 0.0912, but Hindi and long-category 5% guards **fail**. The overall improvement does not establish Task 2 eligibility. Audio sanity statistics match the saved per-item CSV. Initial duplicate-run logs are preserved in the fetched archive; timings from that run are not used to claim a bonus speed-up.

The submitted idea card remains unchanged. The post-midterm pronunciation/timing runner and bonus test are implemented and CPU-tested, **not yet measured on the GPU**. See [POSTMIDTERM.md](POSTMIDTERM.md) for the source-only server update, exact execution/fetch commands, ablations, timing exclusions, loudness diagnostic, and writing still needed after the experiments. No Task 3 or Task 4 scores are invented.

Exact commands run for this update:

```bash
.venv-task1/bin/python code/import_gpu_results.py server_results/task2_validation/25B1207_tts_bootcamp.zip
.venv-task1/bin/python -m unittest discover -s code -p 'test_*.py' -v
.venv-task1/bin/python code/run_postmidterm.py --stage prepare
.venv-task1/bin/python code/build_midterm.py
.venv-task1/bin/python check_format.py 25B1207_tts_bootcamp.zip
.venv-task1/bin/python code/prepare_postmidterm_update.py
```

The 19 CPU checks include a complete synthetic-audio/fake-model run through validation and blind evaluation, plus rule composition, unsupported-word fallback, bounded speed, loudness-only attenuation, ablation controls, a duplicate-run lock and quality gates. This tests implementation and artifact contracts; it is not speech-quality evidence.

## Task 1 reproduction

Run these commands from the kit root. The provided `metrics.py` and baseline recordings are unchanged. Task 1 downloads Whisper and ECAPA for scoring; it does not download IndicF5.

The original organizer kit is required locally. When reproducing from the GitHub checkout, put its `baseline/`, `data/`, `metrics.py`, `check_format.py`, and `results_template.json` at the checkout root first. Keep the kit data untracked. These original kit files and downloaded model weights are not included in the submission. Copies of the completed Task 1 reports are submitted under `code/task1_evidence/`; the commands below regenerate their working copies under `reports/task1/`.

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

The planned Task 3 method combines contextual English pronunciation rewriting with syllable-based generation timing after Task 2 normalization. The experiment will compare normalization alone, rewriting alone, timing alone, and their combination using the same validation items, prompts, and seeds. No Task 3 experiment has been run; all performance fields remain null. `results.json` records the planned dependency as `_planned_task3.builds_on_task2: true`; the unrun result field `task3.builds_on_task2` remains null at midterm because the checker otherwise requires control audio immediately. Set that result field to true when submitting Task 3 audio, together with the required normalization-only `predictions/val_control/`. The midterm idea card is the root `idea.md`; copy its parts (1)–(3) exactly into the form and leave them unchanged after submission.

```bash
.venv-task1/bin/python code/build_midterm.py
.venv-task1/bin/python check_format.py 25B1207_tts_bootcamp.zip
```

The build copies completed scripts, an output-cleared listening notebook, scoring/listening evidence, predictions, results, and the participant's `idea.md` once supplied. It creates `25B1207_tts_bootcamp.zip` and extracts the same contents to `submission/25B1207-tts-bootcamp/` for publishing. Notebook outputs are cleared in the submitted copy because audio playback output can embed the organizer's recordings. The local working notebook retains its playback outputs.

The submitted scoring log replaces private local folder paths with `<PROJECT>` or `<USER_HOME>`. Scores, transcripts, and listening judgments are preserved.

For a Task 1-only submission, omit unfinished tasks and use null for their measurements. Create a zip named `<rollno>_tts_bootcamp.zip` containing the submission files, then run `check_format.py` on that zip before uploading. Do not include `.venv-task1/` or downloaded model caches in the zip. Task 3 idea and report files, when attempted, must be written in your own words.

## Task 2: text normalization

The text part is complete. `normalize.py` implements handwritten Hindi number, rupee/paise, date, percentage, and abbreviation rules. It uses no model or external dictionary. Dates use day/month/year; zero-padded identifiers retain their digits as spoken words. Indian magnitude words stay attached to their amounts before `रुपये`. Uppercase Latin runs are assumed to be initialisms, motivated by RBI/ATM/GST in validation; ordinary mixed/lowercase English words keep their original spelling. This assumption can misclassify an ordinary English word written entirely in capitals.

Rules were developed and checked using validation only, then frozen by SHA256 before applying them programmatically to evaluation. Evaluation text was not displayed or used to add rules. `predictions/t2_normalized.csv` contains all 64 rows. All eight hard validation texts match an accepted spoken reference after the kit's spelling/punctuation folding; the other 16 original texts are unchanged. **These are text checks, not measured audio CER improvements.**

Exact local commands run:

```bash
.venv-task1/bin/python code/test_normalize.py
.venv-task1/bin/python code/prepare_task2.py
.venv-task1/bin/python code/test_task2_pipeline.py
.venv-task1/bin/python check_format.py .
```

Evidence is in `reports/task2/rules_frozen.json` and `validation_text_checks.csv`, copied to `code/task2_evidence/` by the submission builder. The rule fingerprint and normalized-CSV fingerprint must match before synthesis.

## Task 2: initial lab audio workflow

The participant ran the validation audio stage on their lab GPU and fetched its measurements, summarized above. The commands below document that initial setup; use the post-midterm update for the next experiment. The midterm form has been submitted; parts (1)–(3) of `idea.md` are frozen.

### Lab GPU terminal workflow

Transfer only `25B1207_tts_gpu_input.zip` from the Mac to the server; it contains the original kit plus current participant work (about 9.3 MB), without weights, caches or the Mac environment. Replace the example host/directory with your own:

```bash
# On the Mac, from the project folder:
scp 25B1207_tts_gpu_input.zip YOUR_USER@YOUR_SERVER:/YOUR_WORK_DIRECTORY/

# On the server, inside that work directory:
mkdir -p tts_bootcamp
unzip 25B1207_tts_gpu_input.zip -d tts_bootcamp
cd tts_bootcamp
python3 code/server_preflight.py
```

Extract the input bundle into a new folder. After an experiment, rerun the CLI in that existing folder; do not extract the starter bundle over measured results. `server_preflight.py` prints the Python/PyTorch versions, GPU model, driver and whether CUDA is available to Python. It downloads nothing and prints no credentials. Run in the GPU allocation your lab provides, rather than assuming a login shell has GPU access.

The reported lab setup is Python 3.12.3, NVIDIA RTX PRO 4500 Blackwell (32 GB), driver 580.178.04, with no Torch installation. `setup_lab_gpu.sh` creates `.venv-tts` and installs matching Torch/TorchAudio 2.7.1 and TorchVision 0.22.1 from the official CUDA 12.8 wheel index. PyTorch 2.7 introduced Blackwell support in that build. [Official release](https://pytorch.org/blog/pytorch-2-7/), [installation commands](https://pytorch.org/get-started/previous-versions/).

The original kit's install file has two problems: its GitHub source revision is actually the Hugging Face model revision (the GitHub API returns “No commit found”), and its NumPy>=2.1 requirement conflicts with the official source's NumPy<=1.26.4. The supplied files are preserved. Our separate `requirements_gpu_server.txt` uses verified GitHub source revision `13f7c4d627cc10111aea8fe9c0039462cacacdc7`, NumPy 1.26.4 and compatible dependency pins on Python 3.10–3.12. [Official source dependency declaration](https://github.com/ai4bharat/IndicF5/blob/13f7c4d627cc10111aea8fe9c0039462cacacdc7/setup.py).

This is an installation repair. The model revision in `baseline/config.yaml` remains `fb652a96e508091045890dae5bb99fe511facbcf`; the original wrapper, length patch, prompts and decoding settings are unchanged. The repaired package list resolved successfully in a local Python 3.12 pip dry run; actual CUDA imports and synthesis must still be checked on the lab machine.

The participant's server subsequently passed the CUDA calculation with Torch 2.7.1+cu128. Its first import check exposed a transitive dependency issue: unpinned PyArrow 26 requires NumPy 2 at runtime despite pip reporting no broken requirements. PyArrow is now pinned to 19.0.1, compatible with the NumPy 1.26.4 used by IndicF5. A fresh isolated local environment installed the full repaired requirements and passed NumPy/Arrow array conversion plus F5-TTS, Transformers and SpeechBrain imports; no weights were loaded or synthesis performed. The lab runtime check must still be rerun. [Arrow 19 installation requirements](https://arrow.apache.org/docs/19.0/python/install.html). To repair an existing server environment, run the commands below before continuing to Hugging Face login and validation:

```bash
source .venv-tts/bin/activate
python -m pip install numpy==1.26.4 pyarrow==19.0.1
python -m pip check
python code/server_preflight.py --check-runtime
```

On the server, from the extracted project folder:

```bash
bash code/setup_lab_gpu.sh
source .venv-tts/bin/activate
python -c 'from huggingface_hub import login; login(add_to_git_credential=False)'
python code/run_task2.py --stage val
```

Accept the IndicF5 model terms on Hugging Face before login, and paste a read-access token into its hidden prompt. Never place it in a script or shell command. Setup verifies a CUDA calculation, the supplied F5 length patch and the synthesis/scoring imports without loading model weights. The runner also verifies gated model-config access before downloading scoring weights. Its Python module cache stays at the location required by the kit's patcher; weight downloads can use a separate `HF_HUB_CACHE`. Review `reports/task2/comparison.json` and the validation clips, then continue:

```bash
python code/run_task2.py --stage eval
python code/run_task2.py --stage package
```

Bring back `25B1207_tts_bootcamp.zip` after the run. It holds the completed predictions, actual `results.json` and scoring evidence without model weights or kit data. Keep the server's working folder for Task 3 controls and later timing; model downloads are reused. The lab commands above are prepared instructions, not a claim that server synthesis has been executed.

The participant's updated repository overview is preserved in `code/repository_overview.md`, which the builder uses for the root `README.md`. Its Task 2 pending sentence is refreshed automatically after actual audio measurements. Update other completed/pending descriptions as work proceeds, while retaining the submitted idea-card wording.

### Optional Colab/Kaggle workflow

The local Mac has no CUDA GPU and Task 1's environment does not include IndicF5. `code/task2_run.ipynb` is an optional notebook interface for a Python 3.10–3.12 GPU runtime compatible with the CUDA 12.8 wheels. It stops on Python 3.13 instead of attempting the incompatible NumPy installation. Put the original kit and these participant files together in one folder, including `reports/task2/rules_frozen.json` from the working project or its packaged copy at `code/task2_evidence/rules_frozen.json`. The runner restores and verifies the packaged fingerprint when needed. Select the folder as `KIT` and accept the gated IndicF5 model terms. Supply a read-only Hugging Face token through Colab Secrets (`HF_TOKEN`) or the notebook's hidden prompt; never put a token in code, logs or the repository.

For an easier transfer, these local commands create a separate GPU input bundle:

```bash
.venv-task1/bin/python code/build_midterm.py
.venv-task1/bin/python code/prepare_gpu_bundle.py
```

Upload `25B1207_tts_gpu_input.zip` to the top level of your own Google Drive (`MyDrive`). The notebook extracts it into `MyDrive/25B1207_tts_gpu_input` only if that folder does not exist, so later sessions retain your generated audio and results. For Kaggle, extract/copy it to your writable `KIT` folder. This convenience ZIP includes the original kit data; **it is not the submission ZIP and must not be pushed to the public repository**. Neither ZIP contains downloaded model weights or the virtual environment. The submission builder includes only the prescribed submission files.

The notebook uses the repaired dependency file described above. Restart the runtime if installation changes NumPy and then continue from the authentication/check cell. The following are the prepared reproduction commands once the GPU environment has been installed; no GPU measurements are claimed until they have actually completed:

```bash
python code/run_task2.py --stage val
python code/run_task2.py --stage eval
python code/run_task2.py --stage package
```

`--stage val` rescores the provided baseline recordings in the GPU environment, synthesizes all 24 normalized validation items into `predictions/val_control/`, then scores them. Model revision, assigned prompts, seed and all decoding settings stay unchanged from the baseline. `record_task2_results.py` verifies the complete per-item scores before filling Task 2 CER/SIM/hard CER in `results.json`, and records the 5% CER guards for other categories in `reports/task2/comparison.json`. These are development scores; final scoring uses different recognizers.

Review validation scores and the two validation playback examples. If needed, develop further using validation only. When continuing development after applying a previous frozen version blindly, do not inspect evaluation inputs or use evaluation outputs to choose rules. Regenerate the frozen text artifact with the new validation-developed rules before starting a new run.

`--stage eval` generates exactly 40 Task 2 evaluation WAVs with normalization only. It displays no evaluation text and provides no evaluation playback. Audio is checked in a staging folder before moving it into predictions; previous complete audio is preserved outside the submission. Never choose or tune using evaluation listening. `--stage package` rebuilds the same ZIP name and runs the format checker. GPU evidence is archived under `code/task2_evidence/`; staging and backup audio are excluded.

Task 3 has not run. The participant confirmed the existing idea card parts (1)–(3) were submitted to the midterm form; they remain unchanged. Its normalization-only control will be the Task 2 validation audio. Task 4 is an optional later step. `report.md` and idea-card part (4) require actual experiment results and the participant's own wording.

AI assistance: OpenAI Codex assisted with environment setup, reproduction commands, listening notebook preparation, recording the participant's supplied labels, Task 2 normalization and checks, the GPU workflow, and submission packaging. Error observations are the participant's listening judgments. Idea-card and report wording must be the participant's own.
