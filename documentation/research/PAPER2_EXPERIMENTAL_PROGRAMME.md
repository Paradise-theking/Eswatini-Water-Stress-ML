# Paper 2 — Experimental programme and decision framework

Paper 2 explores event-oriented and state-transition forecasting beyond Paper 1's continuous SWBA forecast. All experiments are exploratory, isolated from production, and must be evaluated chronologically. The frozen test period must not be used to tune models, thresholds, features or decision rules.

## Experimental modules
1. **Binary drought state and onset** — forecast SWBA <= -1.0 and onset, horizons 1/3/6 months. Balanced logistic regression versus training-fold prevalence and seasonally smoothed probability baseline. See `PAPER2_PHASE2_EVENT_FORECASTING.md`.
2. **Three-state future classification** — Normal, Watch, Drought at horizons 1/3/6. Balanced multinomial logistic regression versus seasonal month-conditioned class distribution. See `PAPER2_TRANSITION_FORECASTING.md`.
3. **Directional transition forecasting** — forecast improvement/no change/deterioration at horizons 1/3/6 based on origin and future state severity. Balanced multinomial logistic regression versus month-conditioned directional rates. See `PAPER2_DIRECTION_FORECASTING.md`.

## Common safeguards
- Expanding chronological validation; no random shuffling.
- Fold-local preprocessing, fitting, and baseline estimation.
- Class counts and skipped folds reported.
- Probability metrics emphasized alongside classification metrics.
- No claims of operational readiness from exploratory cross-validation alone.
- Results are not verified until workflows have run and artifacts have been inspected.

## Planned research after these experiments
1. Verify all runs and inspect fold outputs, not just aggregate means.
2. Compare experiments on matched chronological validation periods and define a common benchmark table.
3. Assess calibration and threshold-independent probability skill; do not assume class-balanced probabilities are calibrated.
4. Examine persistence and transition-specific conditional skill with sufficient sample counts.
5. Consider ordinal/state transition formulations or regularized Bayesian approaches only if they are justified by validation evidence.
6. Investigate event severity and duration models only with defensible sample-size and censoring treatment.
7. Reserve a genuinely untouched final evaluation period and document all decisions made before it.

## Publication framing
The paper should present methodological development, comparative evidence, limitations, and potential decision relevance. Avoid claims of predicting rare extreme events unless supported by out-of-sample event counts, uncertainty intervals, and appropriate baselines.
