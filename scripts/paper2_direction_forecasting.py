"""Ordinal drought transition-direction experiment for Paper 2.
Predict deterioration / no-change / improvement in state severity from t to t+h.
State labels: normal=0, watch=1, drought=2. Direction labels are based on
future state minus origin state; uses origin and climate predictors at t.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"data/processed/expanded/mnjoli_water_stress_monthly_1981_2025_swba.csv"
OUT=ROOT/"outputs/research/paper2_direction_forecasting"
HORIZONS=[1,3,6]
N_FOLDS=5
MIN_TRAIN=120
LABELS=[-1,0,1] # improvement, no change, deterioration
LABEL_NAMES={-1:"improvement",0:"no_change",1:"deterioration"}

def prepare():
    d=pd.read_csv(SOURCE,parse_dates=["month_date"]).sort_values("month_date").set_index("month_date").asfreq("MS")
    swba=pd.to_numeric(d.swba_3month,errors="coerce")
    state=pd.Series(np.nan,index=d.index)
    state.loc[swba.notna() & (swba>-0.5)]=0
    state.loc[swba.notna() & (swba<=-0.5) & (swba>-1.0)]=1
    state.loc[swba.notna() & (swba<=-1.0)]=2
    cols=[c for c in ["precipitation_mm","pet_mm","water_balance_mm","temperature_mean_c",
      "temperature_min_c","temperature_max_c","dewpoint_c","surface_pressure",
      "u_component_of_wind_10m","v_component_of_wind_10m",
      "surface_solar_radiation_downwards_sum","volumetric_soil_water_layer_1",
      "volumetric_soil_water_layer_2","runoff_sum","surface_runoff_sum",
      "water_balance_3month","water_balance_6month"] if c in d.columns]
    X=d[cols].copy()
    for c in cols:
        X[c+"_lag1"]=d[c].shift(1); X[c+"_lag3"]=d[c].shift(3)
    X["origin_state"]=state
    X["swba_origin"]=swba
    for lag in [1,3,6]:X["swba_lag"+str(lag)]=swba.shift(lag)
    X["month_sin"]=np.sin(2*np.pi*d.index.month/12)
    X["month_cos"]=np.cos(2*np.pi*d.index.month/12)
    return d,X,state

def folds(n):
    if n<=MIN_TRAIN:return []
    start=min(max(MIN_TRAIN,int(n*.60)),n-1)
    b=np.linspace(start,n,N_FOLDS+1,dtype=int)
    return [(np.arange(b[i]),np.arange(b[i],b[i+1])) for i in range(N_FOLDS) if b[i+1]>b[i]]

def scores(y,p):
    pred=np.array(LABELS)[p.argmax(axis=1)]
    y=np.asarray(y,dtype=int)
    return {"accuracy":float(accuracy_score(y,pred)),
      "macro_f1":float(f1_score(y,pred,labels=LABELS,average="macro",zero_division=0)),
      "log_loss":float(log_loss(y,p,labels=LABELS)),
      "brier":float(np.mean(np.sum((np.eye(3)[[LABELS.index(v) for v in y]]-p)**2,axis=1)))}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    d,X,state=prepare(); records=[]
    for h in HORIZONS:
        future=state.shift(-h)
        target=(future-state).rename("direction")
        dat=pd.concat([X,target],axis=1).dropna(subset=["direction"])
        # exclude invalid feature rows only by imputation in model; keep alignment
        dat["direction"]=dat.direction.astype(int)
        xx=dat.drop(columns="direction"); yy=dat.direction
        for fold,(tr,va) in enumerate(folds(len(dat)),1):
            yt,yv=yy.iloc[tr],yy.iloc[va]
            dates_t,dates_v=dat.index[tr],dat.index[va]
            meta={"horizon_months":h,"fold":fold,"train_start":str(dates_t[0].date()),
              "train_end":str(dates_t[-1].date()),"validation_start":str(dates_v[0].date()),
              "validation_end":str(dates_v[-1].date()),"train_n":len(yt),"validation_n":len(yv),
              "train_counts":{LABEL_NAMES[k]:int((yt==k).sum()) for k in LABELS},
              "validation_counts":{LABEL_NAMES[k]:int((yv==k).sum()) for k in LABELS}}
            # Smoothed seasonal transition-direction probabilities.
            tab=pd.DataFrame({"y":yt.values,"month":dates_t.month})
            cnt=tab.groupby("month").y.value_counts().unstack(fill_value=0).reindex(columns=LABELS,fill_value=0)
            ps=[]
            for m in dates_v.month:
                row=cnt.loc[m].values if m in cnt.index else np.zeros(3)
                ps.append((row+1)/(row.sum()+3))
            ps=np.asarray(ps)
            rec={**meta,"model":"seasonal_direction_rate","status":"ok",**scores(yv,ps)}
            records.append(rec)
            if yt.nunique()<2:
                records.append({**meta,"model":"multinomial_direction_model","status":"skipped_single_class_training",
                  "accuracy":None,"macro_f1":None,"log_loss":None,"brier":None})
                continue
            model=Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler()),
              ("clf",LogisticRegression(C=1.0,max_iter=3000,class_weight="balanced"))])
            model.fit(xx.iloc[tr],yt)
            raw=model.predict_proba(xx.iloc[va])
            p=np.zeros((len(yv),3))
            for j,k in enumerate(model.named_steps["clf"].classes_):p[:,LABELS.index(int(k))]=raw[:,j]
            records.append({**meta,"model":"multinomial_direction_model","status":"ok",**scores(yv,p)})
    res=pd.DataFrame(records);res.to_csv(OUT/"fold_scores.csv",index=False)
    good=res[res.status=="ok"]
    summary=good.groupby(["horizon_months","model"]).agg(folds=("fold","count"),
      mean_accuracy=("accuracy","mean"),mean_macro_f1=("macro_f1","mean"),
      mean_log_loss=("log_loss","mean"),mean_brier=("brier","mean")).reset_index()
    summary.to_csv(OUT/"summary.csv",index=False)
    (OUT/"metadata.json").write_text(json.dumps({"direction_definition":"future state minus origin state; positive=deterioration, negative=improvement",
      "states":{"normal":0,"watch":1,"drought":2},"horizons_months":HORIZONS,
      "models":["balanced multinomial logistic regression","seasonal direction-rate baseline"],
      "validation":"expanding chronological folds","no_final_test_used":True,
      "caution":"Directional outcomes collapse transitions; they do not distinguish transition types with identical direction. Exploratory."},indent=2))
    print(summary.to_string(index=False))
if __name__=="__main__":main()
