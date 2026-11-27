"""Matched-origin Paper 2 diagnostic predictions and fold metrics."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score,average_precision_score,brier_score_loss,log_loss,confusion_matrix
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"data/processed/expanded/mnjoli_water_stress_monthly_1981_2025_swba.csv"
OUT=ROOT/"outputs/research/paper2_matched_diagnostics"
H=[1,3,6]; K=5; STATES=[0,1,2]; DIRS=[-1,0,1]
def prep():
 d=pd.read_csv(SRC,parse_dates=["month_date"]).sort_values("month_date").set_index("month_date").asfreq("MS")
 s=pd.to_numeric(d.swba_3month,errors="coerce")
 st=pd.Series(np.nan,index=d.index); st[s.notna()&(s>-.5)]=0;st[s.notna()&(s<=-.5)&(s>-1)]=1;st[s.notna()&(s<=-1)]=2
 cols=[c for c in ["precipitation_mm","pet_mm","water_balance_mm","temperature_mean_c","temperature_min_c","temperature_max_c","dewpoint_c","surface_pressure","u_component_of_wind_10m","v_component_of_wind_10m","surface_solar_radiation_downwards_sum","volumetric_soil_water_layer_1","volumetric_soil_water_layer_2","runoff_sum","surface_runoff_sum","water_balance_3month","water_balance_6month"] if c in d.columns]
 X=d[cols].copy()
 for c in cols:X[c+"_lag1"]=d[c].shift(1);X[c+"_lag3"]=d[c].shift(3)
 X["swba_origin"]=s;X["origin_state"]=st
 for lag in [1,3,6]:X["swba_lag"+str(lag)]=s.shift(lag)
 X["month_sin"]=np.sin(2*np.pi*d.index.month/12);X["month_cos"]=np.cos(2*np.pi*d.index.month/12)
 return d,s,st,X
def model():
 return Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler()),("clf",LogisticRegression(C=1,max_iter=3000,class_weight="balanced"))])
def folds(n):
 start=int(n*.60); b=np.linspace(start,n,K+1,dtype=int)
 return [(np.arange(b[i]),np.arange(b[i+1])) for i in range(K) if b[i+1]>b[i]]
def main():
 OUT.mkdir(parents=True,exist_ok=True);d,s,st,X=prep();rows=[];scores=[]
 for h in H:
  future=s.shift(-h);fstate=st.shift(-h)
  onset=((future<=-1)&(s.shift(-(h-1))>-1)).astype(float);onset[future.isna()|s.shift(-(h-1)).isna()]=np.nan
  binary=(future<=-1).astype(float);binary[future.isna()]=np.nan
  tasks={"drought_state":binary,"onset":onset,"multistate":fstate,"direction":np.sign(fstate-st)}
  for fold,(ti,vi) in enumerate(folds(len(d)),1):
   for task,y in tasks.items():
    valid=y.notna();tr=ti[valid.iloc[ti].to_numpy()];va=vi[valid.iloc[vi].to_numpy()]
    if not len(tr) or not len(va):continue
    yt=y.iloc[tr].astype(int);yv=y.iloc[va].astype(int)
    if yt.nunique()<2: continue
    m=model();m.fit(X.iloc[tr],yt);raw=m.predict_proba(X.iloc[va]);classes=m.named_steps["clf"].classes_
    meta={"task":task,"horizon":h,"fold":fold}
    if task in ("drought_state","onset"):
     p=raw[:,list(classes).index(1)] if 1 in classes else np.zeros(len(va));pred=(p>=.5).astype(int)
     for j,idx in enumerate(va):rows.append({**meta,"origin":d.index[idx].date().isoformat(),"target_date":d.index[idx+h].date().isoformat(),"actual":int(yv.iloc[j]),"predicted":int(pred[j]),"probability":float(p[j])})
     tn,fp,fn,tp=confusion_matrix(yv,pred,labels=[0,1]).ravel()
     scores.append({**meta,"n":len(va),"events":int(yv.sum()),"tp":int(tp),"fp":int(fp),"fn":int(fn),"tn":int(tn),"precision":precision_score(yv,pred,zero_division=0),"recall":recall_score(yv,pred,zero_division=0),"f1":f1_score(yv,pred,zero_division=0),"false_positive_rate":fp/(fp+tn) if tn+fp else None,"pr_auc":average_precision_score(yv,p) if yv.sum()>0 else None,"brier":brier_score_loss(yv,p)})
    else:
     labels=STATES if task=="multistate" else DIRS;full=np.zeros((len(va),3))
     for j,c in enumerate(classes):full[:,labels.index(int(c))]=raw[:,j]
     pred=np.array(labels)[full.argmax(axis=1)]
     for j,idx in enumerate(va):rows.append({**meta,"origin":d.index[idx].date().isoformat(),"target_date":d.index[idx+h].date().isoformat(),"actual":int(yv.iloc[j]),"predicted":int(pred[j]),"probabilities":json.dumps(full[j].tolist())})
     scores.append({**meta,"n":len(va),"accuracy":accuracy_score(yv,pred),"macro_f1":f1_score(yv,pred,labels=labels,average="macro",zero_division=0),"log_loss":log_loss(yv,full,labels=labels),"brier":float(np.mean(np.sum((np.eye(3)[[labels.index(int(v)) for v in yv]]-full)**2,axis=1)))})
 pd.DataFrame(rows).to_csv(OUT/"matched_validation_predictions.csv",index=False);sc=pd.DataFrame(scores);sc.to_csv(OUT/"matched_fold_metrics.csv",index=False)
 sc.groupby(["task","horizon"]).mean(numeric_only=True).reset_index().to_csv(OUT/"matched_metric_summary.csv",index=False)
 (OUT/"metadata.json").write_text(json.dumps({"timeline_folds":"common full monthly timeline; 60% initial segment then five contiguous validation blocks","horizons":H,"target":"future SWBA drought state; onset at t+h; 3-state severity and change in severity","pipeline":"fold-local median imputation, standardization, balanced logistic regression","final_test_used":False,"warning":"exploratory, sparse onset counts; class-weighted probabilities are not assumed calibrated"},indent=2))
if __name__=="__main__":main()
