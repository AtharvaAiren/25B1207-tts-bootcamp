# Evidence for your own report wording

These are measurements and prompts, not a submission report.

Q1: Choose two improvements and two regressions; if there are no regressions, choose the two smallest gains.

| Item | Idea CER minus provided-baseline CER |
|---|---:|
| val_hd_01 | -0.523810 |
| val_hd_08 | -0.512821 |
| val_hg_01 | -0.419355 |
| val_hd_05 | -0.408163 |
| val_hd_03 | -0.346939 |
| val_hd_06 | -0.311688 |
| val_hd_07 | -0.311111 |
| val_hg_06 | -0.295455 |
| val_hd_02 | -0.267857 |
| val_hg_05 | -0.260870 |
| val_hg_03 | -0.260000 |
| val_hd_04 | -0.254902 |
| val_hg_04 | -0.210526 |
| val_hg_02 | -0.108108 |
| val_hi_04 | -0.016667 |
| val_hi_06 | -0.013514 |
| val_hi_01 | -0.012987 |
| val_hi_05 | +0.000000 |
| val_lg_01 | +0.000000 |
| val_lg_02 | +0.003802 |
| val_lg_04 | +0.017483 |
| val_hi_02 | +0.019608 |
| val_lg_03 | +0.022472 |
| val_hi_03 | +0.030769 |

Q2: Compare rewrite, timing, and combined scores. Did timing help beyond rewriting? What would you change?
Q3: Pick one next change and explain what measured failure motivates it.

Also write idea.md part (4) in at most 18 words after the (4) marker; preserve parts (1)–(3) exactly.

The problem statement requires idea.md and report.md in your own words. Each report answer is at most 75 words.

Uniform loudness attenuation is shared by the idea variants and fast method. Compare control_loudness with control to isolate the clipping effect; compare combined with control_loudness to assess pronunciation/timing with the same loudness treatment.
