# State-conditioned transition baseline

This experiment estimates the future state probability distribution conditional on the state observed at the forecast origin. The transition frequencies are estimated exclusively from each training fold and Laplace-smoothed. For horizons 1, 3, and 6 months, the future categorical state (Normal, Watch, Drought) is compared with a balanced multinomial logistic regression on matched expanding validation folds.

Metrics: accuracy, macro-F1, multiclass log-loss, and Brier score. Sparse transitions may make the baseline unstable, particularly at longer horizons and for the drought state. It is exploratory, and the frozen final test period is not used for tuning.

Script: `scripts/paper2_conditional_transition_matrix.py`
Outputs after execution: `outputs/research/paper2_conditional_transitions/fold_scores.csv`, `summary.csv`, `metadata.json`.
