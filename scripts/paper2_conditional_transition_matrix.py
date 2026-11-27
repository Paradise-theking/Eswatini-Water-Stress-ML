"""Paper 2 conditional transition matrix and origin-state persistence baseline.
Predict future categorical state conditional on current state, with monthly
transition matrices estimated strictly from training data; compares logistic
multinomial and transition-probability baseline on the same folds.
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
OUT=ROOT/"outputs/research/paper2_conditional_transitions"
HORIZONS=[1,3,6]
MIN_TRAIN=120
FOLDS=5
STATES=[0,1,2]
NAMES={0:"normal",1:"watch",2:"drought"}

def load():
    d=pd.read_csv(SOURCE,parse_dates=["month_date"]).sort_values("month_date").set_index("month_date").asfreq("MS")
    s=pd.to_numeric(d.swba_3month,errors="coerce")
    state=pd.Series(np.nan,index=d.index)
    state.loc[s.notna()&(s>-0.5)]=0
    state.loc[s.notna()&(s<=-0.5)&(s>-1)]=1
    state.loc[s.notna()&(s<=-1)]=2
    cols=[c for c in ["precipitation_mm","pet_mm","water_balance_mm","temperature_mean_c","temperature_min_c",
      "temperature_max_c","dewpoint_c","surface_pressure","u_component_of_wind_10m","v_component_of_wind_10m",
      "surface_solar_radiation_downwards_sum","volumetric_soil_water_layer_1","volumetric_soil_water_layer_2",
      "runoff_sum","surface_runoff_sum","water_balance_3month","water_balance_6month"] if c in d.columns]
    X=d[cols].copy()
    for c in cols:X[c+"_lag1"]=d[c].shift(1);X[c+"_lag3"]=d[c].shift(3)
    X["origin_state"]=state;X["swba_origin"]=s
    for lag in [1,3,6]:X["swba_lag"+str(lag)]=s.shift(lag)
    X["month_sin"]=np.sin(2*np.pi*d.index.month/12);X["month_cos"]=np.cos(2*np.pi*d.index.month/12)
    return X,state

def make_folds(n):
    if n<=MIN_TRAIN:return []
    start=min(max(MIN_TRAIN,int(.6*n)),n-1)
    b=np.linspace(start,n,FOLDS+1,dtype=int)
    return [(np.arange(b[i]),np.arange(b[i],b[i+1])) for i in range(FOLDS) if b[i+1]>b[i]]

def brier(y,p):
    onehot=np.eye(3)[np.asarray(y,dtype=int)]
    return float(np.mean(np.sum((onehot-p)**2,axis=1)))

def metric(y,p):
    pred=p.argmax(axis=1)
    return {"accuracy":float(accuracy_score(y,pred)),
      "macro_f1":float(f1_score(y,pred,labels=STATES,average="macro",zero_division=0)),
      "log_loss":float(log_loss(y,p,labels=STATES)),"brier":brier(y,p)}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    X,state=load();records=[]
    for h in HORIZONS:
        target=state.shift(-h).rename("target")
        origin=state.rename("origin")
        dat=pd.concat([X,target,origin],axis=1).dropna(subset=["target","origin"])
        xx=dat.drop(columns=["target","origin"]);yy=dat.target.astype(int);oo=dat.origin.astype(int)
        for fold,(tr,va) in enumerate(make_folds(len(dat)),1):
            yt,yv=yy.iloc[tr],yy.iloc[va];ot,ov=oo.iloc[tr],oo.iloc[va]
            dates=dat.index
            meta={"horizon_months":h,"fold":fold,"train_start":str(dates[tr[0]].date()),
              "train_end":str(dates[tr[-1]].date()),"validation_start":str(dates[va[0]].date()),
              "validation_end":str(dates[va[-1]].date()),"train_n":len(tr),"validation_n":len(va)}
            # Origin-state transition frequency pooled over horizons in train-fold.
            trans=pd.crosstab(ot,yt).reindex(index=STATES,columns=STATES,fill_value=0)
            probs=[]
            for st in ov:
                probs.append((trans.loc[st].values+1)/(trans.loc[st].sum()+3))
            p=np.array(probs)
            records.append({**meta,"model":"origin_state_transition_rate","status":"ok",**metric(yv,p)})
            if yt.nunique()<2:
                records.append({**meta,"model":"multinomial_logistic","status":"skipped_single_class",
                  "accuracy":None,"macro_f1":None,"log_loss":None,"brier":None})
                continue
            model=Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler()),
             ("clf",LogisticRegression(C=1,max_iter=3000,class_weight="balanced"))])
            model.fit(xx.iloc[tr],yt)
            raw=model.predict_proba(xx.iloc[va]);p2=np.zeros((len(yv),3))
            for j,k in enumerate(model.named_steps["clf"].classes_):p2[:,int(k)]=raw[:,j]
            records.append({**meta,"model":"multinomial_logistic","status":"ok",**metric(yv,p2)})
    res=pd.DataFrame(records);res.to_csv(OUT/"fold_scores.csv",index=False)
    good=res[res.status=="ok"]
    summary=good.groupby(["horizon_months","model"]).agg(folds=("fold","count"),
      mean_accuracy=("accuracy","mean"),mean_macro_f1=("macro_f1","mean"),
      mean_log_loss=("log_loss","mean"),mean_brier=("brier","mean")).reset_index()
    summary.to_csv(OUT/"summary.csv",index=False)
    (OUT/"metadata.json").write_text(json.dumps({"states":NAMES,"horizons":HORIZONS,
      "baseline":"training-fold transition matrix conditioned on forecast-origin state, Laplace smoothing",
      "model":"balanced multinomial logistic regression C=1","validation":"expanding folds",
      "no_final_test_used":True,"warning":"Transition matrices are estimated from sparse fold-local samples."},indent=2))
    print(summary.to_string(index=False))
if __name__=="__main__":main()
