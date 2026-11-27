"""Paper 2 exploratory multi-state drought transition forecasting.
Three SWBA states: normal (> -0.5), watch (-1.0 < SWBA <= -0.5),
drought (SWBA <= -1.0). Targets future states at 1, 3, and 6 months.
Chronological rolling validation; no final test-period tuning.
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
OUT=ROOT/"outputs/research/paper2_transition_forecasting"
HORIZONS=[1,3,6]
N_FOLDS=5
MIN_TRAIN=120
CLASSES=[0,1,2]
CLASS_LABELS={0:"normal",1:"watch",2:"drought"}

def load():
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
        X[c+"_lag1"]=d[c].shift(1)
        X[c+"_lag3"]=d[c].shift(3)
    X["swba_current"]=swba
    for lag in [1,3,6]:
        X["swba_lag"+str(lag)]=swba.shift(lag)
    X["month_sin"]=np.sin(2*np.pi*d.index.month/12)
    X["month_cos"]=np.cos(2*np.pi*d.index.month/12)
    return d,X,state

def folds(n):
    if n<=MIN_TRAIN:return []
    initial=min(max(MIN_TRAIN,int(n*.60)),n-1)
    b=np.linspace(initial,n,N_FOLDS+1,dtype=int)
    return [(np.arange(0,b[i]),np.arange(b[i],b[i+1])) for i in range(N_FOLDS) if b[i+1]>b[i]]

def multiclass_brier(y,probs):
    onehot=np.eye(3)[np.asarray(y,dtype=int)]
    return float(np.mean(np.sum((probs-onehot)**2,axis=1)))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    d,X,state=load()
    allrows=[]
    for h in HORIZONS:
        target=state.shift(-h).rename("target")
        dat=pd.concat([X,target],axis=1).dropna(subset=["target"])
        xx=dat.drop(columns="target")
        yy=dat.target.astype(int)
        for fold,(tr,va) in enumerate(folds(len(dat)),1):
            xt,xv=xx.iloc[tr],xx.iloc[va]
            yt,yv=yy.iloc[tr],yy.iloc[va]
            dates_t,dates_v=dat.index[tr],dat.index[va]
            meta={"horizon_months":h,"fold":fold,"train_start":str(dates_t[0].date()),
              "train_end":str(dates_t[-1].date()),"validation_start":str(dates_v[0].date()),
              "validation_end":str(dates_v[-1].date()),"train_n":len(yt),"validation_n":len(yv),
              "train_class_counts":{CLASS_LABELS[k]:int((yt==k).sum()) for k in CLASSES},
              "validation_class_counts":{CLASS_LABELS[k]:int((yv==k).sum()) for k in CLASSES}}
            # Seasonal baseline: Laplace-smoothed class distribution by calendar month.
            train_tab=pd.DataFrame({"y":yt.values,"month":dates_t.month})
            counts=train_tab.groupby("month").y.value_counts().unstack(fill_value=0).reindex(columns=CLASSES,fill_value=0)
            probs=[]
            for m in dates_v.month:
                row=counts.loc[m].values if m in counts.index else np.zeros(3)
                probs.append((row+1)/(row.sum()+3))
            pseason=np.asarray(probs)
            predseason=pseason.argmax(axis=1)
            allrows.append({**meta,"model":"seasonal_month_distribution","accuracy":accuracy_score(yv,predseason),
              "macro_f1":f1_score(yv,predseason,labels=CLASSES,average="macro",zero_division=0),
              "log_loss":log_loss(yv,pseason,labels=CLASSES),"multiclass_brier":multiclass_brier(yv,pseason),"status":"ok"})
            if yt.nunique()<2:
                allrows.append({**meta,"model":"multinomial_logistic","accuracy":None,"macro_f1":None,
                  "log_loss":None,"multiclass_brier":None,"status":"skipped_single_class_training"})
                continue
            model=Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler()),
              ("classifier",LogisticRegression(C=1.0,max_iter=3000,class_weight="balanced"))])
            model.fit(xt,yt)
            raw=model.predict_proba(xv)
            # Align probability columns to the complete 3-state class order.
            p=np.zeros((len(yv),3))
            for j,k in enumerate(model.named_steps["classifier"].classes_):p[:,int(k)]=raw[:,j]
            pred=p.argmax(axis=1)
            allrows.append({**meta,"model":"multinomial_logistic","accuracy":accuracy_score(yv,pred),
              "macro_f1":f1_score(yv,pred,labels=CLASSES,average="macro",zero_division=0),
              "log_loss":log_loss(yv,p,labels=CLASSES),"multiclass_brier":multiclass_brier(yv,p),"status":"ok"})
    res=pd.DataFrame(allrows)
    res.to_csv(OUT/"fold_scores.csv",index=False)
    good=res[res.status=="ok"]
    summary=good.groupby(["horizon_months","model"]).agg(folds=("fold","count"),
      mean_accuracy=("accuracy","mean"),mean_macro_f1=("macro_f1","mean"),
      mean_log_loss=("log_loss","mean"),mean_brier=("multiclass_brier","mean")).reset_index()
    summary.to_csv(OUT/"summary.csv",index=False)
    meta={"states":{"normal":"> -0.5","watch":"<= -0.5 and > -1.0","drought":"<= -1.0"},
      "forecast_horizons_months":HORIZONS,"validation":"expanding sequential validation",
      "baseline":"training-fold month-of-year categorical distribution with Laplace smoothing",
      "model":"balanced multinomial logistic regression C=1","no_final_test_used":True,
      "warning":"Exploratory; class imbalance and limited drought samples require cautious interpretation."}
    (OUT/"metadata.json").write_text(json.dumps(meta,indent=2))
    print(summary.to_string(index=False))
if __name__=="__main__":main()
