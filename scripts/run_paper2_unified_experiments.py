"""Run the Paper 2 event/state/transition experiments in one reproducible sequence."""
from pathlib import Path
import subprocess,sys,json,datetime
ROOT=Path(__file__).resolve().parents[1]
SCRIPTS=["paper2_event_forecasting.py","paper2_transition_forecasting.py",
 "paper2_direction_forecasting.py","paper2_conditional_transition_matrix.py"]
OUT=ROOT/"outputs/research/paper2_unified_run"
OUT.mkdir(parents=True,exist_ok=True)
status=[]
for script in SCRIPTS:
    started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    p=subprocess.run([sys.executable,str(ROOT/"scripts"/script)],cwd=ROOT,text=True,capture_output=True)
    entry={"script":script,"started_utc":started,"exit_code":p.returncode,"stdout":p.stdout,"stderr":p.stderr}
    status.append(entry)
    (OUT/(script.replace(".py",".log"))).write_text(p.stdout+"\n--- STDERR ---\n"+p.stderr)
    print(f"{script}: exit={p.returncode}")
    if p.returncode:
        print(p.stderr)
(OUT/"run_manifest.json").write_text(json.dumps({"run_time_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"scripts":status},indent=2))
if any(x["exit_code"] for x in status):
    sys.exit(1)
