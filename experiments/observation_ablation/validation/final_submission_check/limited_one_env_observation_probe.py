import os,sys,json,time,traceback,hashlib,importlib,collections
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path('/home/zxro/ant_rl_submission');OUT=ROOT/'experiments/observation_ablation/validation/final_submission_check'
name,task,folder,width,iteration=sys.argv[1:];width=int(width);iteration=int(iteration)
stem='runtime_obs_'+name+'_lerobot_arena';app=None;env=None;stage='app_initialization';calls=collections.Counter()
def ts():return datetime.now(timezone.utc).isoformat()
def write(name,data):
 with (OUT/name).open('w') as f:json.dump(data,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def event(kind,**kw):
 row={'event':kind,'timestamp':ts(),'monotonic':time.monotonic(),**kw}
 with (OUT/(stem+'_events.jsonl')).open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
 print(json.dumps(row),flush=True)
def profile(frame,event,arg):
 if event=='call':
  file=frame.f_code.co_filename;fun=frame.f_code.co_name
  if ('simulation_context.py' in file or 'simulation_manager.py' in file) and fun in ('step','reset','forward','render','_warm_start','warm_start'):
   calls[file+':'+fun]+=1
result={'timestamp':ts(),'python':sys.executable,'conda_environment':os.environ.get('CONDA_DEFAULT_ENV'),'isaac_lab_version':(ROOT/'VERSION').read_text().strip(),'task_id':task,'num_envs':1,'checkpoint_path':str(ROOT/'experiments/observation_ablation'/folder/'checkpoints/best_model.pt'),'checkpoint_selected_iteration':iteration,'expected_config_dimension':width,'actual_runtime_observation_shape':None,'result':'FAIL','reset_occurred':False,'rollout_steps':0,'shutdown_result':'PENDING','process_exit_code':None}
try:
 from isaaclab.app import AppLauncher
 app=AppLauncher(headless=True).app;event('app_initialized')
 stage='imports'
 import importlib.metadata as metadata
 result['isaac_sim_version']=metadata.version('isaacsim')
 import isaaclab,isaaclab_tasks,ant,gymnasium as gym,torch
 assert str(Path(isaaclab.__file__).resolve()).startswith(str(ROOT/'source'))
 assert str(Path(ant.__file__).resolve()).startswith(str(ROOT/'source'))
 result['import_paths']={'isaaclab':isaaclab.__file__,'ant':ant.__file__}
 stage='config_load'
 spec=gym.spec(task);entry=spec.kwargs['env_cfg_entry_point'];module,attr=entry.split(':');cfg=getattr(importlib.import_module(module),attr)()
 result['config_class']=entry;manifest=json.loads((ROOT/'experiments/observation_ablation'/folder/'manifest.json').read_text())
 result['preserved_manifest_observation']=manifest['observation'];assert manifest['observation']['actor']==width
 cfg.scene.num_envs=1;cfg.seed=manifest['training_seed'];result['environment_seed']=cfg.seed
 result['terrain_seed']=cfg.scene.terrain.terrain_generator.seed
 assert result['terrain_seed']==manifest['terrain_seed']
 stage='checkpoint_load'
 ckpt=torch.load(result['checkpoint_path'],map_location='cpu',weights_only=True);sd=ckpt['model_state_dict'];actor=sd['actor.0.weight'];critic=sd['critic.0.weight']
 result.update(checkpoint_actor_input_dimension=int(actor.shape[1]),checkpoint_actor_first_layer_shape=list(actor.shape),checkpoint_critic_input_dimension=int(critic.shape[1]),checkpoint_saved_iteration=int(ckpt['iter']))
 result['checkpoint_tensors_finite']=all(bool(torch.isfinite(t).all()) for t in sd.values() if torch.is_tensor(t))
 assert result['checkpoint_saved_iteration']==iteration
 stage='environment_creation_scene_sensor_manager_initialization';event('environment_creation_started');sys.setprofile(profile)
 env=gym.make(task,cfg=cfg);base=env.unwrapped;assert base.num_envs==1;event('environment_created')
 stage='reset_observation_computation';event('reset_invoked');result['reset_occurred']=True
 obs,_=env.reset(seed=cfg.seed);event('reset_returned');tensor=obs['policy'];manager=base.observation_manager
 sys.setprofile(None)
 result.update(actual_runtime_observation_shape=list(tensor.shape),dtype=str(tensor.dtype),device=str(tensor.device),finite=bool(torch.isfinite(tensor).all()),nan_count=int(torch.isnan(tensor).sum()),inf_count=int(torch.isinf(tensor).sum()),observation_term_order=manager.active_terms['policy'],observation_term_dimensions=manager.group_obs_term_dim['policy'],manager_group_observation_shape=manager.group_obs_dim['policy'],simulation_python_call_counts=dict(calls),environment_control_step_counter=int(base._sim_step_counter),simulation_step_occurred=any(k.endswith(':step') and v for k,v in calls.items()),simulation_step_count_scope='Python simulation_context step calls, including initialization internals; native physics calls are not separately counted',reward_terms=base.reward_manager.active_terms)
 result['triple_matches']=tensor.shape==(1,width) and result['checkpoint_actor_input_dimension']==width and result['checkpoint_critic_input_dimension']==width
 result['source_manifest_order_matches_runtime']=result['observation_term_order']==manifest['observation']['order']
 result['result']='PASS' if result['triple_matches'] and result['finite'] and result['checkpoint_tensors_finite'] and result['source_manifest_order_matches_runtime'] else 'FAIL'
 result['evidence_timestamp']=ts();write(stem+'.json',result);event('observation_evidence_flushed',result=result['result'],shape=result['actual_runtime_observation_shape'])
except BaseException:
 sys.setprofile(None);result.update(failure_stage=stage,traceback=traceback.format_exc(),simulation_python_call_counts=dict(calls));write(stem+'.json',result);event('validation_failed',stage=stage,traceback=result['traceback'])
finally:
 if env is not None:
  event('env_close_invoked')
  try:env.close();event('env_close_returned')
  except BaseException:event('env_close_exception',traceback=traceback.format_exc())
 if app is not None:
  event('app_close_invoked')
  try:app.close();event('app_close_returned')
  except BaseException:event('app_close_exception',traceback=traceback.format_exc())
 event('process_finished')
