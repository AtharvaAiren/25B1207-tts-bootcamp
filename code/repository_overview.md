# Hindi/Hinglish TTS Evaluation and Text Normalisation

Inter-IIT TTS bootcamp submission for **25B1207**. The work covers baseline evaluation, participant listening annotations, handwritten text and pronunciation rules, a syllable-timing experiment, and a measured speed test. No model weights are trained.

[Portfolio](https://github.com/AtharvaAiren) · [Reproduction guide](code/README.md) · [Recorded results](results.json)

## Completed work

- Scored 24 supplied baseline recordings: character error rate (CER) **0.2426**, speaker similarity (SIM) **0.8303**, with zero failed items. These are baseline measurements, not improvements from a new model.
- Reviewed all 14 recordings above CER 0.20 and recorded participant listening judgments.
- Implemented text-normalisation rules for Hindi numbers, currency, dates, percentages and abbreviations.
- Prepared all 64 normalised texts and a GPU workflow for synthesis and scoring.
- Completed the pronunciation/timing ablations on all 24 validation items and generated 40 evaluation clips each for Tasks 2, 3 and 4.
- Verified all 168 submitted generated WAVs against their recorded hashes and checked both validation folders against saved sanity statistics and CER transcripts.

## Measured validation results

| System | Overall CER | Hinglish CER | SIM |
| --- | ---: | ---: | ---: |
| Supplied baseline, rescored on GPU | 0.2421 | 0.3020 | 0.8303 |
| Normalisation only | 0.1616 | 0.2986 | 0.8078 |
| Pronunciation rules only | 0.0535 | 0.0283 | 0.8455 |
| Timing only | 0.1279 | 0.3262 | 0.8446 |
| Submitted combination (Task 3) | 0.0571 | 0.0429 | 0.8421 |
| Faster combination (Task 4) | 0.0636 | 0.0580 | 0.8410 |

Task 3 lowers Hinglish CER by **85.79%** relative to the supplied baseline and passes its development guards. Rewriting alone performs better than the combination, so the predicted benefit from timing is not supported. The predeclared combination remains submitted. Loudness safety is shared by the idea variants; a separate loudness-matched control isolates clipping effects.

Task 4 measures **1.868×** speed-up: baseline **137.14 s**, fast **73.42 s**, on the same GPU in the same session with loading and one warm-up excluded. Its CER rises **11.38%**, within the allowed 20%.

Task 2 reduces hard-category CER from **0.4583 to 0.0912**, but its Hindi/long guards against supplied audio fail. The fresh same-session baseline and normalisation control produce identical audio for all 16 unchanged-text items, showing the difference was already present in baseline reproduction. These development findings do not establish evaluation scores; evaluation audio was not listened to or used for tuning.

## Submission status

All technical and written deliverables are included. `report.md` contains the three report answers, and `idea.md` includes the result in part (4). Idea parts (1)–(3) are unchanged from the submitted midterm prediction. The latest format check passes.

## Explore the repository

| Resource | Purpose |
| --- | --- |
| [code/normalize.py](code/normalize.py) | Handwritten normalisation rules |
| [code/task1_evidence/](code/task1_evidence/) | Baseline scoring and listening evidence |
| [code/task2_evidence/](code/task2_evidence/) | Frozen-rule evidence |
| [code/task2_run.ipynb](code/task2_run.ipynb) | Prepared GPU synthesis workflow |
| [code/postmidterm_evidence/comparison.json](code/postmidterm_evidence/comparison.json) | Measured controls, ablations and guards |
| [code/task3_review.ipynb](code/task3_review.ipynb) | Validation-only report examples and playback |
| [code/baseline_timing.json](code/baseline_timing.json), [code/fast_timing.json](code/fast_timing.json) | Same-session bonus timing evidence |
| [code/README.md](code/README.md) | Setup, reproduction commands and limitations |

## Reproduction prerequisites

The original organiser kit is required locally. Its recordings, data, scoring scripts and model weights are excluded from this public repository. Follow the reproduction guide for Python 3.12 setup and the GPU-stage prerequisites.
