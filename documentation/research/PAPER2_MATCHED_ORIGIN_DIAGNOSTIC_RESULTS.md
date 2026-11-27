# Paper 2 matched-origin diagnostic results (exploratory)

**Run:** GitHub Actions run [45991995037](https://github.com/Paradise-theking/Eswatini-Water-Stress-ML/actions/runs/45991995037)  
**Commit:** `c66fee76b2cfc367a76765b667a2c78a518ea6d5`  
**Artifact:** `paper2-matched-diagnostics`  
**Status:** successful execution; results are exploratory validation, not final held-out test evidence.

## Validation design

The corrected script uses five non-overlapping chronological validation blocks over the final 40% of the 1981–2025 monthly timeline, with expanding training windows. For horizon (h), training origins are retained only when their target month is strictly earlier than the first validation origin. Imputation, scaling and balanced logistic regression are fitted within each fold.

The common validation origins begin in January 2008 and extend through December 2025 (with horizon-specific missing terminal labels). The run generated 215, 213 and 210 predictions for the 1-, 3- and 6-month horizons, respectively.

## Main binary warning results

The primary drought-state threshold is SWBA <= -1. The onset target is a new transition into drought at exactly the target month: SWBA(t+h) <= -1 while SWBA(t+h-1) > -1. Predictions use a provisional probability threshold of 0.5.

| Task | Horizon | N | Observed positives | TP | FP | FN | Precision | Recall | PR-AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Drought state | 1 month | 215 | 27 | 24 | 23 | 3 | 0.511 | 0.889 | 0.699 | 0.088 |
| Drought state | 3 months | 213 | 27 | 12 | 56 | 15 | 0.176 | 0.444 | 0.166 | 0.226 |
| Drought state | 6 months | 210 | 27 | 10 | 58 | 17 | 0.147 | 0.370 | 0.152 | 0.225 |
| Drought onset | 1 month | 215 | 11 | 7 | 44 | 4 | 0.137 | 0.636 | 0.134 | 0.153 |
| Drought onset | 3 months | 213 | 11 | 2 | 64 | 9 | 0.030 | 0.182 | 0.050 | 0.230 |
| Drought onset | 6 months | 210 | 11 | 4 | 69 | 7 | 0.055 | 0.364 | 0.066 | 0.212 |

## Initial interpretation

1. One-month drought-state classification is materially stronger than the longer horizons in this run: recall is 0.889 and precision is 0.511 at the provisional 0.5 threshold.
2. This does not translate into reliable onset warning. At one month, 7 of 11 onset cases are detected, but there are 44 false onset alerts, giving precision of only 0.137.
3. Onset performance deteriorates at three months (2/11 detected; 64 false alerts) and remains weak at six months (4/11 detected; 69 false alerts).
4. The gap between drought-state detection and onset prediction suggests that recognizing an already-developing/persistent drought state is easier than predicting the transition into drought.
5. The balanced classifier's probabilities are not calibrated by assumption. The 0.5 threshold is provisional and should not be presented as an operational warning threshold.

## Scope and limitations

- This run contains model predictions only; it does not yet provide a baseline comparison on identical origin dates.
- Earlier event, seasonal and transition-matrix experiment artifacts used different validation windows and/or target constructions. Their summary metrics must not be compared directly with this run.
- The 11 onset observations are a small event sample. Event-level uncertainty is substantial.
- The current-origin-month predictors imply issuance after that month's observations are available.
- No final held-out test set was used and no production model was changed.

## Next required experiment

Implement persistence, training-fold prevalence, calendar-month onset climatology, and fold-local conditional state-transition probabilities inside the same matched-origin loop. Save baseline predictions for the exact same validation origins and target rows as the logistic model. Then compare event hits, false alarms, PR-AUC, Brier score and event-level coverage using common denominators. Only after that comparison should the central Paper 2 contribution be selected.
