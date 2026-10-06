"""Passive evidence collector; executes the canonical evaluator without changing its logic."""
import os,sys,json,time,runpy
from pathlib import Path
from datetime import datetime,timezone
script=Path(sys.argv[1]).resolve();out=Path(sys.argv[sys.argv.index('--output')+1]).resolve();record_dir=out.parent
sys.argv=[str(script),*sys.argv[2:]];sys.path.insert(0,str(script.parent))
seen=set()
def record(event,**values):
 row={'event':event,'timestamp':datetime.now(timezone.utc).isoformat(),'monotonic':time.monotonic(),**values}
 with (record_dir/'runtime_events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
 print('[VALIDATION] '+json.dumps(row),flush=True)
def profile(frame,event,arg):
 code=frame.f_code;name=code.co_name;file=code.co_filename
 if event=='return' and name=='get_observations' and file.endswith('rsl_rl/vecenv_wrapper.py') and 'observation' not in seen:
  seen.add('observation');t=arg['policy'];record('policy_observation',shape=list(t.shape),dtype=str(t.dtype),device=str(t.device))
 if event=='return' and name=='load' and file.endswith('runners/on_policy_runner.py') and 'checkpoint' not in seen:
  seen.add('checkpoint');runner=frame.f_locals['self'];record('checkpoint_loaded',actor_first_layer_shape=list(runner.alg.policy.actor[0].weight.shape),selected_iteration=runner.current_learning_iteration)
 if event in ('call','return') and name=='close' and ('manager_based' in file or 'simulation_app.py' in file):
  kind=('app' if 'simulation_app.py' in file else 'env')+'_close_'+('invoked' if event=='call' else 'returned')
  if kind not in seen:
   seen.add(kind)
   if event=='call' and out.exists():
    for p in out.rglob('*'):
     if p.is_file():
      with p.open('rb') as f:os.fsync(f.fileno())
   record(kind)
record('canonical_evaluator_started',script=str(script),arguments=sys.argv,python=sys.executable)
sys.setprofile(profile)
try:runpy.run_path(str(script),run_name='__main__')
finally:sys.setprofile(None)
record('canonical_evaluator_returned')
