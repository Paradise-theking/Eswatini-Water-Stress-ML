"""Collect fold summaries from experiment outputs; preserve source score files."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"outputs/research"
FILES=[
("binary",BASE/"paper2_event_forecasting"/"rolling_validation_scores.csv"),
("states",BASE/"paper2_transition_forecasting"/"fold_scores.csv"),
("direction",BASE/"paper2_direction_forecasting"/"fold_scores.csv"),
("conditional",BASE/"paper2_conditional_transitions"/"fold_scores.csv")]
OUT=BASE/"paper2_unified_run"
OUT.mkdir(parents=True,exist_ok=True)
rows=[]
for label,path in FILES:
    if not path.exists():
        rows.append({"experiment":label,"status":"missing_outputs","file":str(path)})
        continue
    df=pd.read_csv(path)
    df.insert(0,"experiment",label)
    rows.append(df)
frames=[r for r in rows if isinstance(r,pd.DataFrame)]
if frames:
    combined=pd.concat(frames,ignore_index=True,sort=False)
    combined.to_csv(OUT/"all_fold_results_long.csv",index=False)
    summary_rows=[]
    for exp,df in combined.groupby("experiment"):
        for key in [c for c in ["task","horizon_months","model"] if c in df.columns]:
            pass
        groups=[c for c in ["experiment","task","horizon_months","model"] if c in df.columns]
        numeric=[c for c in ["pr_auc","brier","roc_auc","precision","recall","f1","accuracy","macro_f1","log_loss","multiclass_brier"] if c in df.columns]
        if groups and numeric:
            g=df.groupby(groups,dropna=False)[numeric].mean().reset_index()
            summary_rows.append(g)
    if summary_rows:pd.concat(summary_rows,ignore_index=True,sort=False).to_csv(OUT/"aggregated_means.csv",index=False)
    print("Consolidated",len(combined),"rows.")
else:
    print("No results CSV files found yet.")
