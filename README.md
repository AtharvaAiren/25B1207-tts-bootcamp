# Hindi/Hinglish TTS Evaluation and Text Normalisation

Inter-IIT TTS bootcamp submission for **25B1207**. The work covers evaluation of supplied speech recordings, participant listening annotations, handwritten Hindi/Hinglish text-normalisation rules and a prepared GPU synthesis workflow.

[Portfolio](https://github.com/AtharvaAiren) · [Reproduction guide and AI-assistance disclosure](code/README.md) · [Recorded results](results.json)

## Completed work

- Scored 24 supplied baseline recordings: character error rate (CER) **0.2426**, speaker similarity (SIM) **0.8303**, with zero failed items. These are baseline measurements, not improvements from a new model.
- Reviewed all 14 recordings above CER 0.20 and recorded participant listening judgments.
- Implemented text-normalisation rules for Hindi numbers, currency, dates, percentages and abbreviations.
- Prepared all 64 normalised texts and a GPU workflow for synthesis and scoring.

## Pending work

Task 2 evaluation audio and validation audio scores remain pending. The Task 3 [idea card](idea.md) describes a proposed pronunciation/timing experiment; its predicted improvement is a hypothesis, not a measured result.

## Explore the repository

| Resource | Purpose |
| --- | --- |
| [code/normalize.py](code/normalize.py) | Handwritten normalisation rules |
| [code/task1_evidence/](code/task1_evidence/) | Baseline scoring and listening evidence |
| [code/task2_evidence/](code/task2_evidence/) | Frozen-rule evidence |
| [code/task2_run.ipynb](code/task2_run.ipynb) | Prepared GPU synthesis workflow |
| [code/README.md](code/README.md) | Setup, reproduction commands, limitations and assistance disclosure |

## Reproduction prerequisites

The original organiser kit is required locally. Its recordings, data, scoring scripts and model weights are excluded from this public repository. Follow the reproduction guide for Python 3.12 setup and the GPU-stage prerequisites.
