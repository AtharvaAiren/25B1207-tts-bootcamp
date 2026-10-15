# Post-midterm experiments and results

The GPU run completed, including all 120 evaluation outputs and 48 submitted
validation outputs. Overall validation CER is 0.0571 for the combined Task 3
method, with SIM 0.8421 and Hinglish CER 0.0429. Rewriting alone is better on
Hinglish (0.0283), so the timing-benefit prediction is not supported. Task 4 has
CER 0.0636 and same-session speed-up 1.8680, within the 20% CER guard. See
`code/postmidterm_evidence/comparison.json` in the submission for all ablations.

The frozen source is unchanged. The instructions below reproduce the experiment;
another GPU run is not needed to finish the current written deliverables.

The first normalization run is saved in `server_results/task2_validation/` locally.
Its hard CER improved from 0.4583 to 0.0912 (80.10% relative reduction), but Hindi
and long-category guards failed. One Hindi item failed the clipping sanity check.
The supplied baseline and regenerated audio differ even where text is unchanged;
the fresh same-session raw baseline confirms identical audio to the control for
all 16 unchanged-text items. This diagnoses a reproduction difference rather
than a normalization effect. It does not erase the failed guards against the
official supplied baseline.

## What the runner does

`run_postmidterm.py` implements the submitted idea card without editing it:

1. Generate a fresh raw baseline and a normalization-only control, with seed 0
   and the original pinned wrapper/settings/prompts.
2. Test rewriting alone, timing alone, and their combination on all 24 validation
   items. No word lookup table or additional model is used. English spellings are
   parsed by reusable handwritten patterns; uncovered patterns retain Latin.
3. Test the same combined method at 16 flow steps for the optional bonus.
4. Score the supplied baseline and every generated validation system with the
   unmodified kit metrics. Save per-item CSVs, summaries, hashes and timings.
5. In the evaluation stage, synthesize 40 items each for Task 2 and Task 3. Include
   Task 4 only if it passes CER <= 1.2 times Task 3, <=10% failed items, and is faster.
   The combined Task 3 method is predeclared; an ablation winning does not cause
   an unreported switch. No evaluation text or audio is displayed or used to tune.
6. Package results and run `check_format.py`. Missing report/idea-result text
   remains visibly unfinished until the participant supplies their own wording.

Timing estimates count approximate syllable nuclei instead of characters and
derive a rate from the same processed reference clip/transcript used by the
baseline. Generation `speed` is bounded to [0.75, 1.25] for items with English.
Hindi-only items keep speed 1. This changes synthesis duration, not recorded audio
speed. These estimates were fixed before testing; the results did not support
an aggregate benefit over pronunciation rules alone.

The idea variants and bonus uniformly attenuate peaks above 0.95 before PCM save.
This is permitted loudness normalization. The raw baseline and normalization-only
control retain the kit's original saving behavior. A `control_loudness` diagnostic
scores the exact same control synthesis with the same attenuation before saving,
so clipping improvements can be distinguished from pronunciation/timing gains.
Rewrite/timing/combined share this treatment. No pitch change, filtering, denoising
or waveform time stretching is used.

All timed systems run serially in one Python process, with the model loaded and
one warm-up per system excluded; CUDA is synchronized around timing. Verifiers
run only after timed synthesis is over. Baseline and eligible fast timing files
are also copied to `code/baseline_timing.json` and `code/fast_timing.json`.
The lock rejects a second runner in the same project folder; interrupting the
runner also terminates its worker process group.

## Commands

On the Mac (the update ZIP contains code only, preserving remote measurements):

```bash
cd "$HOME/Desktop/Inter-IIT TTS"
.venv-task1/bin/python code/prepare_postmidterm_update.py
scp 25B1207_tts_postmidterm_update.zip dmlab@10.178.30.102:
ssh dmlab@10.178.30.102
```

On the server, with the previous run finished and its results fetched:

```bash
cd ~/tts_25B1207
unzip -o ~/25B1207_tts_postmidterm_update.zip -d .
source .venv-tts/bin/activate
python -m unittest discover -s code -p 'test_*.py' -v
mkdir -p reports/postmidterm
nohup .venv-tts/bin/python -u code/run_postmidterm.py --stage all \
  > reports/postmidterm/console.log 2>&1 < /dev/null &
printf '%s\n' "$!" > reports/postmidterm/run.pid
tail -f reports/postmidterm/console.log
```

Hugging Face login and model terms from the successful previous run are reused.
No environment reinstall is needed. `all` runs validation, blind evaluation and
packaging sequentially. It retains and reports validation quality regressions;
they are not silently turned into passing guards. Official evaluation scoring
may reject those systems. More than 10% validation failures stops evaluation.

Use `--stage val`, `--stage eval`, and `--stage package` separately if reviewing
validation results before evaluation is preferred. `--stage prepare` is CPU only.
Keep a frozen plan and the exact source used for a run. Do not replace the code
of a running experiment or reuse a plan after changing its method.

Monitor in another server terminal:

```bash
cd ~/tts_25B1207
tail -f reports/postmidterm/val_worker.log
# During evaluation instead:
tail -f reports/postmidterm/eval_worker.log
# GPU usage, when needed:
nvidia-smi -l 2
```

Ctrl+C stops a log/GPU monitor only. To stop the experiment itself, run
`kill -TERM "$(cat reports/postmidterm/run.pid)"` in the project folder.

When the runner finishes, on the server:

```bash
cat reports/postmidterm/comparison.json
cat reports/postmidterm/check_format.log
```

Fetch on the Mac:

```bash
cd "$HOME/Desktop/Inter-IIT TTS"
mkdir -p server_results/postmidterm
scp dmlab@10.178.30.102:tts_25B1207/25B1207_tts_bootcamp.zip server_results/postmidterm/
.venv-task1/bin/python code/import_gpu_results.py server_results/postmidterm/25B1207_tts_bootcamp.zip
```

## Final writing

The PS requires `idea.md` and `report.md` in the participant's own words. The
participant explicitly requested drafts and will rewrite them before final
submission. Both files are now populated: report answers have 68/66/71 words,
and the idea has 200 words. Use `code/postmidterm_evidence/writing_notes.md`
inside the ZIP, or its imported copy `reports/postmidterm/writing_notes.md`, as
evidence when rewriting. Keep Q1–Q3 at most 75 words each, and idea part (4) at
at most 18 words after the `(4)` marker (the frozen first three parts already
total 181; the marker also counts as a word). Do not edit
the first three parts. Then rebuild, check and push the final submission.

AI coding assistance: OpenAI Codex wrote the experiment runner, rule implementation,
tests and packaging support with the participant's direction. Tests on CPU check
the controls and implementation; they do not demonstrate audio quality gains.
