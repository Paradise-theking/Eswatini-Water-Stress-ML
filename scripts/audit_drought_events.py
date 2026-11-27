"""
Paper 2: descriptive drought-event audit of expanded monthly SWBA source.
Read-only with respect to source data. Writes reproducible audit outputs only.
"""
from pathlib import Path
import hashlib
import json
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "processed" / "expanded" / "mnjoli_water_stress_monthly_1981_2025_swba.csv"
OUT = ROOT / "outputs" / "research" / "drought_event_audit"
THRESHOLDS = [-0.5, -1.0, -1.5, -2.0]

def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def identify_columns(df):
    date_candidates = [c for c in df.columns if c.lower() in {"date","month","time","datetime","forecast_date","target_date"}]
    swba_candidates = [c for c in df.columns if "swba" in c.lower() and ("target" not in c.lower() and "forecast" not in c.lower())]
    if not date_candidates:
        date_candidates = [c for c in df.columns if "date" in c.lower()]
    if not swba_candidates:
        swba_candidates = [c for c in df.columns if c.lower() in {"water_balance_anomaly","standardized_water_balance_anomaly"}]
    if not date_candidates or not swba_candidates:
        raise ValueError(f"Could not automatically identify date/SWBA columns. Columns={list(df.columns)}")
    return date_candidates[0], swba_candidates[0]

def month_season(m):
    return {12:"DJF",1:"DJF",2:"DJF",3:"MAM",4:"MAM",5:"MAM",6:"JJA",7:"JJA",8:"JJA",9:"SON",10:"SON",11:"SON"}[m]

def episodes_for_threshold(df, threshold, value_col="swba"):
    valid = df[value_col].notna()
    dry = valid & (df[value_col] <= threshold)
    # Consecutive calendar months; explicitly break runs on missing values or month gaps.
    starts = dry & (~dry.shift(1, fill_value=False) | (df["date"].diff().dt.days > 40))
    episode_id = starts.cumsum().where(dry)
    records = []
    for eid, group in df.loc[dry].assign(_eid=episode_id[dry]).groupby("_eid"):
        vals = group[value_col]
        records.append({
            "threshold": threshold,
            "episode_id": int(eid),
            "start_date": group["date"].iloc[0].strftime("%Y-%m"),
            "end_date": group["date"].iloc[-1].strftime("%Y-%m"),
            "duration_months": int(len(group)),
            "minimum_swba": float(vals.min()),
            "mean_swba": float(vals.mean()),
            "deficit_sum_below_threshold": float((threshold - vals).clip(lower=0).sum()),
            "start_season": month_season(group["date"].iloc[0].month),
            "single_month_episode": bool(len(group)==1),
        })
    return pd.DataFrame(records), dry

def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Source dataset not found: {SOURCE}")
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(SOURCE)
    date_col, swba_col = identify_columns(raw)
    df = raw[[date_col, swba_col]].copy()
    df.columns = ["date_raw", "swba_raw"]
    df["date"] = pd.to_datetime(df["date_raw"], errors="coerce")
    df["swba"] = pd.to_numeric(df["swba_raw"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
    duplicate_dates = int(df.duplicated("date").sum())
    if duplicate_dates:
        raise ValueError(f"Duplicate monthly dates detected: {duplicate_dates}. Resolve before audit.")
    full_months = pd.date_range(df.date.min(), df.date.max(), freq="MS")
    missing_months = full_months.difference(df.date)
    df["season"] = df["date"].dt.month.map(month_season)
    df["year"] = df["date"].dt.year
    df["dry_1"] = df["swba"] <= -1.0
    df["severe_2"] = df["swba"] <= -2.0
    df["previous_swba"] = df["swba"].shift(1)
    df["onset_dry_1"] = (df["swba"] <= -1.0) & (df["previous_swba"] > -1.0)
    df["onset_severe_2"] = (df["swba"] <= -2.0) & (df["previous_swba"] > -2.0)
    df.loc[df["previous_swba"].isna(), ["onset_dry_1","onset_severe_2"]] = np.nan

    episode_frames=[]
    threshold_rows=[]
    for thr in THRESHOLDS:
        episodes, dry = episodes_for_threshold(df, thr)
        if not episodes.empty: episode_frames.append(episodes)
        threshold_rows.append({
            "threshold":thr,
            "valid_swba_months":int(df["swba"].notna().sum()),
            "dry_months":int(dry.sum()),
            "dry_month_fraction":float(dry.sum()/df["swba"].notna().sum()),
            "episode_count":int(len(episodes)),
            "single_month_episodes":int(episodes["single_month_episode"].sum()) if len(episodes) else 0,
            "mean_episode_duration":float(episodes["duration_months"].mean()) if len(episodes) else np.nan,
            "max_episode_duration":int(episodes["duration_months"].max()) if len(episodes) else 0,
        })
    episodes_all=pd.concat(episode_frames,ignore_index=True) if episode_frames else pd.DataFrame()
    threshold_summary=pd.DataFrame(threshold_rows)

    # Primary (-1) episode table
    primary=episodes_all.loc[episodes_all["threshold"]==-1.0].copy() if not episodes_all.empty else pd.DataFrame()
    # Annual/seasonal counts
    annual=df.groupby("year").agg(valid_months=("swba","count"),dry_months=("dry_1","sum"),severe_months=("severe_2","sum"),dry_onsets=("onset_dry_1","sum"),severe_onsets=("onset_severe_2","sum")).reset_index()
    seasonal=df.groupby("season").agg(valid_months=("swba","count"),dry_months=("dry_1","sum"),severe_months=("severe_2","sum"),dry_onsets=("onset_dry_1","sum")).reset_index()

    event_labels=df[["date","swba","season","dry_1","severe_2","onset_dry_1","onset_severe_2"]].copy()
    event_labels.to_csv(OUT/"monthly_event_labels.csv",index=False)
    episodes_all.to_csv(OUT/"episode_inventory_all_thresholds.csv",index=False)
    primary.to_csv(OUT/"episode_inventory_threshold_minus1.csv",index=False)
    threshold_summary.to_csv(OUT/"threshold_sensitivity.csv",index=False)
    annual.to_csv(OUT/"annual_event_summary.csv",index=False)
    seasonal.to_csv(OUT/"seasonal_event_summary.csv",index=False)

    metadata={
      "source":str(SOURCE.relative_to(ROOT)),
      "source_sha256":sha256_file(SOURCE),
      "rows_in_source":int(len(raw)),
      "date_column":date_col,"swba_column":swba_col,
      "audit_rows_after_date_parse":int(len(df)),
      "date_start":df.date.min().strftime("%Y-%m"),
      "date_end":df.date.max().strftime("%Y-%m"),
      "duplicate_months":duplicate_dates,
      "missing_calendar_months":[d.strftime("%Y-%m") for d in missing_months],
      "missing_swba_values":int(df.swba.isna().sum()),
      "thresholds":THRESHOLDS,
      "definitions":"Dry state at or below threshold; episode is maximal contiguous monthly run. Missing/gapped months break episodes. Onset defined by transition from previous month above threshold; unknown previous value is not labeled negative.",
    }
    (OUT/"audit_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print("="*72)
    print("PAPER 2 — DESCRIPTIVE DROUGHT EVENT AUDIT")
    print("="*72)
    print(json.dumps(metadata,indent=2))
    print("\nThreshold sensitivity:")
    print(threshold_summary.to_string(index=False))
    print("\nPrimary threshold (-1.0) episode inventory:")
    print(primary.to_string(index=False))
    print(f"\nOutputs written to: {OUT}")

if __name__=="__main__":
    main()
