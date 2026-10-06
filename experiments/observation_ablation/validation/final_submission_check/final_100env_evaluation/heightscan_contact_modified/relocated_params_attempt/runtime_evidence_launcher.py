"""Passive evidence collection around the unchanged dedicated canonical evaluator."""
import sys,os,json,runpy,time
from pathlib import Path
from datetime import datetime,timezone
script=Path(sys.argv[1]).resolve();out=Path(sys.argv[sys.argv.index('--output')+1]).resolve();record_dir=out.parent
sys.argv=[str(script),*sys.argv[2:]];sys.path.insert(0,str(script.parent));seen=set()
def record(event,**data):
 row={'event':event,'timestamp':datetime.now(timezone.utc).isoformat(),'monotonic':time.monotonic(),**data}
 with (record_dir/'runtime_events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
 print('[VALIDATION] '+json.dumps(row),flush=True)
def profile(frame,event,arg):
 name=frame.f_code.co_name;file=frame.f_code.co_filename
 try:
  if event=='return' and name=='get_observations' and file.endswith('rsl_rl/vecenv_wrapper.py') and 'obs' not in seen:
   seen.add('obs');t=arg['policy'];record('policy_observation',shape=list(t.shape),dtype=str(t.dtype),device=str(t.device))
  if event=='return' and name=='load' and file.endswith('runners/on_policy_runner.py') and 'checkpoint' not in seen:
   seen.add('checkpoint');r=frame.f_locals['self'];record('checkpoint_loaded',actor_first_layer_shape=list(r.alg.policy.actor[0].weight.shape),selected_iteration=r.current_learning_iteration)
  if event in ('call','return') and name=='close' and ('manager_based' in file or 'simulation_app.py' in file):
   kind=('app' if 'simulation_app.py' in file else 'env')+'_close_'+('invoked' if event=='call' else 'returned')
   if kind not in seen:
    seen.add(kind)
    if event=='call' and out.exists():
     for p in out.rglob('*'):
      if p.is_file():
       with p.open('rb') as f:os.fsync(f.fileno())
    record(kind)
 except Exception as e:record('collector_error',error=repr(e))
def audit(event,args):
 if event=='import' and args[0]=='ant' and 'rearmed' not in seen:
  seen.add('rearmed');sys.setprofile(profile);record('collector_rearmed_after_app_startup')
sys.addaudithook(audit)
record('canonical_evaluator_started',script=str(script),arguments=sys.argv,python=sys.executable)
sys.setprofile(profile)
try:runpy.run_path(str(script),run_name='__main__')
finally:sys.setprofile(None)
record('canonical_evaluator_returned')
