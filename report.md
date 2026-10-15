Q1
On val_hg_01 and val_hg_06, CER fell from 45.16% to 3.23% and 31.82% to 2.27%. Rewriting helped recover words such as internet, movie and shopping in the transcripts. CER rose from 0% to 3.08% on val_hi_03 and 5.24% to 7.49% on val_lg_03. Their text was unchanged; the same errors appeared in the fresh baseline, so these regressions reflect baseline reproduction differences.

Q2
These tests suggest English spelling mattered more than my timing estimate. Hinglish CER was 30.20% for the supplied baseline, 2.83% with rewriting, and 4.29% with rewriting plus timing. Timing alone reached 32.62%. The combined method beat my predicted 27.2%, but did not beat rewriting alone. Loudness normalization also removed one clipping failure, so I would separate that benefit from pronunciation gains and drop blanket timing changes.

Q3
I would make timing changes conditional instead of applying them to every sentence containing English. On val_hg_02 and val_hg_04, adding timing increased CER compared with rewriting alone, while val_hg_03 improved. I would use validation experiments to find a reliable condition for changing timing, keep normal timing otherwise, and test the rule on validation items not used to choose it before applying it blindly to evaluation.
