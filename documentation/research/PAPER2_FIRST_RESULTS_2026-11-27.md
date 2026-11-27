# Paper 2 — First unified experimental results (27 Nov 2026)

## Execution provenance
Unified workflow run 45990055582 completed successfully on research/drought-event-audit after correction of the directional target. All four scripts executed and artifacts were uploaded. Results below are exploratory chronological validation averages across five folds; not a final test set or operational validation.

## Binary forecasting
### Future drought state (mean five-fold)
| Horizon | PR-AUC | Recall | F1 |
|---|---:|---:|---:|
| 1 month | 0.6629 | 0.8933 | 0.5969 |
| 3 months | 0.2272 | 0.3309 | 0.1890 |
| 6 months | 0.1897 | 0.3391 | 0.1822 |

### Onset
| Horizon | PR-AUC | Recall | F1 |
|---|---:|---:|---:|
| 1 month | 0.2566 | 0.7833 | 0.2482 |
| 3 months | 0.0907 | 0.3000 | 0.0461 |
| 6 months | 0.1547 | 0.3167 | 0.1085 |

Event folds contain approximately 43 validation observations and mean 2.2 onset events. Extreme caution: folds are small and event prevalence low. Fold-wise precision, false alarm counts, uncertainty, and calibration require scrutiny. The state task has mean 5.4 events per fold. Do not interpret high recall as operational skill without false-alarm analysis.

## Multi-state forecast
Balanced multinomial Logistic Regression:
| Horizon | Accuracy | Macro-F1 | Log-loss |
|---|---:|---:|---:|
| 1 | 0.6802 | 0.5751 | 0.7465 |
| 3 | 0.4093 | 0.3351 | 1.2274 |
| 6 | 0.4440 | 0.3338 | 1.1117 |

Seasonal month distribution baseline had mean accuracy 0.6568, 0.6558, and 0.6538 at horizons 1, 3, and 6 respectively. The model appears competitive at 1 month but is weak relative to the baseline at longer horizons; compare log-loss/Brier and class-wise errors as well.

## Directional forecasting
| Horizon | Model Macro-F1 | Seasonal baseline Macro-F1 | Model accuracy |
|---|---:|---:|---:|
| 1 | 0.5742 | 0.2731 | 0.6186 |
| 3 | 0.5675 | 0.2192 | 0.5511 |
| 6 | 0.5612 | 0.2293 | 0.5579 |

Macro-F1 is encouraging relative to the baseline, but model accuracy at one month (0.6186) is lower than the baseline (0.6977). This tradeoff requires per-class breakdown and probability calibration. Target sign labels are improvement, no-change, deterioration.

## Conditional transition baseline vs logistic
| Horizon | Baseline accuracy | Logistic accuracy | Baseline log-loss | Logistic log-loss |
|---|---:|---:|---:|---:|
| 1 | 0.7163 | 0.6791 | 0.6899 | 0.7510 |
| 3 | 0.6538 | 0.4065 | 0.8953 | 1.2467 |
| 6 | 0.6519 | 0.4176 | 0.8886 | 1.1255 |

Origin-state baseline accuracy/log-loss are stronger than the logistic results here. The baseline is class-conditioned with smoothed transition counts and must be interpreted in context of state imbalance and matched validation samples.

## Provisional synthesis
- Strongest apparent binary signal is 1-month future-state prediction, but balanced probabilities and false alarm rates need audit.
- Binary onset is unstable and sparse, particularly longer horizons.
- Multi-state direct prediction weakens sharply beyond 1 month.
- Directional prediction retains higher macro-F1 at 3/6 months than direct multi-state, though evaluation is not harmonized and accuracy/probability metrics differ.
- Conditional state-transition baseline is hard to beat and is especially competitive at 3/6 months.
- These results are preliminary, not confirmatory; no claim of skill for severe/extreme drought.

## Required next work
1. Audit metrics and validation alignment; verify shared samples and absence of target leakage.
2. Generate per-fold predictions, confusion matrices, class-specific metrics, precision/false alarms, and PR curves.
3. Re-evaluate all tasks on matched forecast origins/validation periods.
4. Probability calibration and uncertainty intervals, carefully nested/chronological.
5. Compare against naïve persistence, seasonal climatology, training-fold prevalence and origin-state conditional transition baselines.
6. Consider a carefully justified ordinal or state-conditioned model, but avoid further model proliferation until the baseline comparison is resolved.
