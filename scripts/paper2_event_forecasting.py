"""
Paper 2 experimental framework: prospective drought-state/onset forecasting.
This is a NEW modeling experiment, not a rerun of the completed descriptive audit.
Uses monthly SWBA and climate covariates available at forecast origin, with
strictly chronological expanding-window evaluation. No final test tuning.
"""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score, brier_score_loss,
                             precision_score, recall_score, f1_score)

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"data/processed/expanded/mnjoli_water_stress_monthly_1981_2025_swba.csv"
OUT=ROOT/"outputs/research/paper2_event_forecasting"
HORIZONS=[1,3,6]
THRESHOLD=-1.0
MIN_TRAIN=120
VALIDATION_BLOCK=24
N_FOLDS=5

def load_frame():
    df=pd.read_csv(SOURCE,parse_dates=["month_date"]).sort_values("month_date")
    df=df.set_index("month_date").asfreq("MS")
    if df.index.has_duplicates: raise ValueError("Duplicate months.")
    df["swba"]=pd.to_numeric(df["swba_3month"],errors="coerce")
    # Only values known by forecast origin: predictors from t, plus lagged SWBA.
    base=[c for c in ["precipitation_mm","pet_mm","water_balance_mm",
      "temperature_mean_c","temperature_min_c","temperature_max_c","dewpoint_c",
      "surface_pressure","u_component_of_wind_10m","v_component_of_wind_10m",
      "surface_solar_radiation_downwards_sum","volumetric_soil_water_layer_1",
      "volumetric_soil_water_layer_2","runoff_sum","surface_runoff_sum",
      "water_balance_3month","water_balance_6month"] if c in df.columns]
    X=df[base].copy()
    # Lag climate values: values at t are retained, with lag 1 and lag 3.
    for c in base:
        X[c+"_lag1"]=df[c].shift(1)
        X[c+"_lag3"]=df[c].shift(3)
    X["swba_origin"]=df["swba"]
    X["swba_lag1"]=df["swba"].shift(1)
    X["swba_lag3"]=df["swba"].shift(3)
    X["swba_lag6"]=df["swba"].shift(6)
    X["month_sin"]=np.sin(2*np.pi*df.index.month/12)
    X["month_cos"]=np.cos(2*np.pi*df.index.month/12)
    # Target state at future h. Onset definition requires preceding month not dry.
    y_state={}
    y_onset={}
    dry=df["swba"]<=THRESHOLD
    for h in HORIZONS:
        future_dry=dry.shift(-h).astype("float")
        future_dry[df["swba"].shift(-h).isna()]=np.nan
        y_state[h]=future_dry
        # Exact calendar target month t+h, and its prior month t+h-1.
        target=df["swba"].shift(-h)
        prior=df["swba"].shift(-(h-1))
        onset=((target<=THRESHOLD)&(prior>THRESHOLD)).astype(float)
        onset[target.isna()|prior.isna()]=np.nan
        y_onset[h]=onset
    return df,X,y_state,y_onset

def chronological_folds(index, n=N_FOLDS, min_train=MIN_TRAIN, block=VALIDATION_BLOCK):
    # Partition the full timeline into an initial development segment and
    # sequential validation blocks. Every later observation is evaluated once.
    total=len(index)
    if total<=min_train: return []
    first=max(min_train,int(total*0.60))
    first=min(first,total-1)
    remaining=total-first
    boundaries=np.linspace(first,total,n+1,dtype=int)
    folds=[]
    for i in range(n):
        end_train=boundaries[i]
        end_val=boundaries[i+1]
        if end_val<=end_train: continue
        folds.append((np.arange(0,end_train),np.arange(end_train,end_val)))
    return folds

def metrics(y,p):
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=float)
    pred=p>=.5
    out={"n":int(len(y)),"events":int(y.sum()),"event_fraction":float(y.mean()),
         "precision":float(precision_score(y,pred,zero_division=0)),
         "recall":float(recall_score(y,pred,zero_division=0)),
         "f1":float(f1_score(y,pred,zero_division=0)),
         "brier":float(brier_score_loss(y,p))}
    out["pr_auc"]=float(average_precision_score(y,p)) if y.sum()>0 else None
    out["roc_auc"]=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None
    return out

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    df,X,ys,yo=load_frame()
    rows=[]
    for task,targets in [("future_state",ys),("future_onset",yo)]:
      for h,y0 in targets.items():
        dat=pd.concat([X,y0.rename("target")],axis=1).dropna(subset=["target"])
        xx=dat.drop(columns="target"); yy=dat.target.astype(int)
        folds=chronological_folds(dat.index)
        for fold,(tr,va) in enumerate(folds,1):
          train_x=xx.iloc[tr]; train_y=yy.iloc[tr]
          val_x=xx.iloc[va]; val_y=yy.iloc[va]
          dates_train=dat.index[tr]; dates_val=dat.index[va]
          foldmeta={"task":task,"horizon_months":h,"fold":fold,
            "train_start":str(dates_train[0].date()),"train_end":str(dates_train[-1].date()),
            "validation_start":str(dates_val[0].date()),"validation_end":str(dates_val[-1].date()),
            "features":int(xx.shape[1])}
          pprev=np.full(len(va),float(train_y.mean()))
          rows.append({**foldmeta,"model":"train_prevalence","status":"ok",**metrics(val_y,pprev)})
          train_month=pd.DataFrame({"y":train_y.values,"month":dates_train.month})
          counts=train_month.groupby("month").y.agg(["sum","count"])
          pseason=np.array([(counts.loc[m,"sum"]+1)/(counts.loc[m,"count"]+2)
             if m in counts.index else float(train_y.mean()) for m in dates_val.month])
          rows.append({**foldmeta,"model":"seasonal_rate","status":"ok",**metrics(val_y,pseason)})
          if train_y.nunique()<2:
            rows.append({**foldmeta,"model":"balanced_logistic","status":"skipped_single_class",
              "n":int(len(val_y)),"events":int(val_y.sum()),"event_fraction":float(val_y.mean()),
              "precision":None,"recall":None,"f1":None,"brier":None,"pr_auc":None,"roc_auc":None})
            continue
          model=Pipeline([("impute",SimpleImputer(strategy="median")),
             ("scale",StandardScaler()),("clf",LogisticRegression(
               C=1.0,max_iter=2000,class_weight="balanced"))])
          model.fit(train_x,train_y)
          p=model.predict_proba(val_x)[:,1]
          rows.append({**foldmeta,"model":"balanced_logistic","status":"ok",**metrics(val_y,p)})
    results=pd.DataFrame(rows)
    results.to_csv(OUT/"rolling_validation_scores.csv",index=False)
    successful=results[results.status=="ok"]
    summary=(successful.groupby(["task","horizon_months","model"]).agg(
       folds=("fold","count"),mean_pr_auc=("pr_auc","mean"),mean_brier=("brier","mean"),
       mean_precision=("precision","mean"),mean_recall=("recall","mean"),
       mean_f1=("f1","mean"),mean_events=("events","mean"),mean_n=("n","mean")
       ).reset_index())
    summary.to_csv(OUT/"rolling_validation_summary.csv",index=False)
    meta={"source":str(SOURCE.relative_to(ROOT)),"threshold":THRESHOLD,
      "horizons_months":HORIZONS,"validation":"expanding chronological folds",
      "minimum_initial_train_months":MIN_TRAIN,"validation_blocks":"timeline remainder split into sequential blocks",
      "folds_requested":N_FOLDS,"models":["balanced LogisticRegression C=1","training-fold event prevalence",
      "month-of-year event-rate baseline with Laplace smoothing"],
      "probability_decision_threshold":0.5,"no_final_test_used":True,
      "note":"Exploratory rolling validation; sparse events may destabilize fold metrics. Baselines are computed from training folds only."}
    (OUT/"experiment_metadata.json").write_text(json.dumps(meta,indent=2))
    print(summary.to_string(index=False))
    print("Outputs:",OUT)
if __name__=="__main__": main()
