"""Frozen teammate1 v3 depth policy, original host dynamics, common RS stock reward."""
import os,sys,json,hashlib,argparse,csv,types,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RS=Path('/home/zxro/IsaacLab_RS');TEAM2=Path('/home/zxro/teammate_ant_rl_2')
sys.path.insert(0,str(ROOT/'source'));sys.path.insert(0,str(TEAM2))
if os.environ.get('ISAAC_SIM_SITE_PACKAGES'):sys.path.append(os.environ['ISAAC_SIM_SITE_PACKAGES'])
for package in ('isaaclab','isaaclab_assets','isaaclab_tasks','isaaclab_rl'):
    sys.path.insert(0,str(RS/'source'/package))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--environment',choices=['final_unseen','teammate1','teammate2'],required=True)
p.add_argument('--checkpoint',default=str(ROOT/'logs/rsl_rl/ant/v3_depth/best_model.pt'))
p.add_argument('--seed',type=int,default=24);p.add_argument('--num_envs',type=int,default=100)
p.add_argument('--output',required=True,type=Path)
AppLauncher.add_app_launcher_args(p);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
a.output.mkdir(parents=True)
a.enable_cameras=True
app=AppLauncher(a).app
import faulthandler
faulthandler.enable()
faulthandler.dump_traceback_later(90, repeat=True)
import gymnasium as gym
import numpy as np,torch,yaml
from pxr import UsdGeom
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import RewardsCfg
from isaaclab_tasks.manager_based.classic.ant.ant_final_unseen_env_cfg import AntFinalUnseenEnvCfg
from ant.ant_env_cfg import AntEnvCfg as Team1HostCfg,ObservationsCfg as Team1ObservationsCfg
from ant.depth_actor_critic import DepthActorCritic
from ant_rough.env_cfg import AntSelfEvaluationCfg

HOSTS={'final_unseen':AntFinalUnseenEnvCfg,'teammate1':Team1HostCfg,'teammate2':AntSelfEvaluationCfg}
TASKS={'final_unseen':'Isaac-Ant-Team1Depth-FinalUnseen-v0','teammate1':'Isaac-Ant-Team1Depth-Teammate1-v0','teammate2':'Isaac-Ant-Team1Depth-Teammate2-v0'}
ORIGINAL={'final_unseen':'Isaac-Ant-Final-Unseen-v0','teammate1':'Ant-rl-v0','teammate2':'Isaac-Ant-Rough-E8A-SelfEval-v0'}
EXPECTED_SHA='29a9474cc6e6d488b50ca33b9bc7ec99b516fdfc4313c3663d3843535990c2c5'
TERMS=['progress','alive','upright','move_to_target','action_l2','energy','joint_pos_limits']
JOINTS=['front_left_leg','front_right_leg','left_back_leg','right_back_leg','front_left_foot','front_right_foot','left_back_foot','right_back_foot']
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False,default=str)+'\n')
def stats(x):
    x=np.asarray(x,dtype=np.float64)
    return dict(mean=float(x.mean()),std=float(x.std(ddof=0)),min=float(x.min()),max=float(x.max()),median=float(np.median(x)))

def main():
    assert sha(a.checkpoint)==EXPECTED_SHA,'Preselected checkpoint changed'
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    assert checkpoint['iter']==6632
    state=checkpoint['model_state_dict']
    assert tuple(state['actor.0.weight'].shape)==(400,123) and tuple(state['actor.6.weight'].shape)==(8,100)
    original=HOSTS[a.environment]();cfg=copy.deepcopy(original)
    native=Team1HostCfg()
    cfg.observations=Team1ObservationsCfg()
    cfg.scene.depth_camera=copy.deepcopy(native.scene.depth_camera)
    cfg.rewards=RewardsCfg()
    # All geometry/materials/action/reset/termination/physics are inherited without overrides.
    equal={}
    for key in ('terrain','robot'):
        equal['scene.'+key]=getattr(cfg.scene,key).to_dict()==getattr(original.scene,key).to_dict()
    for key in ('sim','actions','events','terminations'):
        equal[key]=getattr(cfg,key).to_dict()==getattr(original,key).to_dict()
    assert all(equal.values()),equal
    assert cfg.observations.to_dict()==native.observations.to_dict()
    assert cfg.scene.depth_camera.to_dict()==native.scene.depth_camera.to_dict()
    cfg.seed=a.seed;cfg.scene.num_envs=a.num_envs;cfg.log_dir=str(a.output)
    if a.device:cfg.sim.device=a.device
    dump(a.output/'config.json',cfg.to_dict())
    gym.register(id=TASKS[a.environment],entry_point='isaaclab.envs:ManagerBasedRLEnv',disable_env_checker=True)
    print('[INIT] creating environment',flush=True)
    env=gym.make(TASKS[a.environment],cfg=cfg);base=env.unwrapped
    print('[INIT] environment created',flush=True)
    agent=yaml.safe_load((ROOT/'logs/rsl_rl/ant/v3_depth/params/agent.yaml').read_text())
    assert agent['policy']['actor_hidden_dims']==agent['policy']['critic_hidden_dims']==[400,200,100]
    assert agent['policy']['activation']=='elu' and not agent['policy']['actor_obs_normalization'] and not agent['policy']['critic_obs_normalization']
    print('[INIT] wrapper initial reset',flush=True)
    env=RslRlVecEnvWrapper(env,clip_actions=agent['clip_actions'])
    print('[INIT] first observations ready',flush=True)
    obs=env.get_observations();n=env.num_envs;dt=base.step_dt;robot=base.scene['robot']
    assert tuple(obs['policy'].shape)==(n,59) and tuple(obs['depth'].shape)==(n,48,64,1)
    assert env.num_actions==8 and robot.joint_names==JOINTS
    assert base.action_manager.get_term('joint_effort')._joint_names==JOINTS
    params=copy.deepcopy(agent['policy']);params.pop('class_name')
    model=DepthActorCritic(obs,agent['obs_groups'],8,**params).to(base.device)
    model.load_state_dict(state,strict=True);model.eval()
    with torch.inference_mode():features=model._actor_input(obs);assert features.shape==(n,123)
    start=robot.data.root_state_w.clone();origins=base.scene.env_origins.clone()
    assignment=dict(rows=base.scene.terrain.terrain_levels.cpu().tolist(),columns=base.scene.terrain.terrain_types.cpu().tolist(),origins=origins.cpu().tolist())
    mesh=UsdGeom.Mesh(base.scene.stage.GetPrimAtPath('/World/ground/terrain/mesh'))
    meshhash=hashlib.sha256(np.asarray(mesh.GetPointsAttr().Get(),dtype=np.float32).tobytes()+np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),dtype=np.int32).tobytes()).hexdigest()
    assert base.reward_manager.active_terms==TERMS
    definitions=[]
    for term in TERMS:
        cfgterm=base.reward_manager.get_term_cfg(term);func=cfgterm.func;owner=func if hasattr(func,'__qualname__') else type(func)
        definitions.append(dict(name=term,function=owner.__module__+'.'+owner.__qualname__,weight=cfgterm.weight,params=cfgterm.params))
    done=torch.zeros(n,dtype=torch.bool,device=base.device);returns=torch.zeros(n,dtype=torch.float64,device=base.device)
    official=torch.zeros(n,device=base.device);steps=torch.zeros(n,dtype=torch.long,device=base.device)
    components=torch.zeros((n,7),dtype=torch.float64,device=base.device)
    velocity=torch.zeros(n,dtype=torch.float64,device=base.device);end=torch.zeros(n,device=base.device);legacy=end.clone();maximum=end.clone()
    terminal_x=end.clone();terminal_v=end.clone();fall=done.clone();timeout=done.clone();boundary=done.clone()
    original_reset=base._reset_idx
    def read_terminal(self,ids):
        terminal_x[ids]=robot.data.root_pos_w[ids,0]-start[ids,0]
        terminal_v[ids]=robot.data.root_lin_vel_w[ids,0]
        return original_reset(ids)
    base._reset_idx=types.MethodType(read_terminal,base)
    fallterm={'final_unseen':'torso_height','teammate1':'body_z_down','teammate2':'body_orientation'}[a.environment]
    max_step_error=0.;sum_step_error=0.;sample_count=0;raw_nonfinite=0;frame_updates=0
    # Native preprocessing accepts raw sky/background Inf; processed depth must be finite in [0,1].
    camera=base.scene['depth_camera'];initial_camera_frames=camera.frame.clone()
    previous_frame=camera.frame.clone();camera_updates=torch.zeros(n,dtype=torch.long,device=base.device)
    depth_min=1.;depth_max=0.;depth_changed=torch.zeros(n,dtype=torch.bool,device=base.device);previous_depth=obs['depth'].clone()
    source_files=[Path(__file__),ROOT/'source/ant/ant_env_cfg.py',ROOT/'source/ant/depth_actor_critic.py',ROOT/'source/ant/depth_obs.py',
        ROOT/'logs/rsl_rl/ant/v3_depth/params/agent.yaml',RS/'source/isaaclab/isaaclab/managers/reward_manager.py',
        RS/'source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/ant_env_cfg.py',
        RS/'source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/ant_final_unseen_env_cfg.py',TEAM2/'ant_rough/env_cfg.py']
    dump(a.output/'source_hashes.json',{str(x):sha(x) for x in source_files})
    output=(a.output/'reward_components_step.csv').open('w',newline='');w=csv.writer(output)
    w.writerow(['env_id','step','time_s','total_reward']+['reward_'+x for x in TERMS]+['done','terminated','truncated'])
    for step in range(base.max_episode_length):
        assert app.is_running(),'Simulator stopped early'
        with torch.inference_mode():
            assert obs['policy'].shape==(n,59) and obs['depth'].shape==(n,48,64,1)
            assert torch.isfinite(obs['policy']).all() and torch.isfinite(obs['depth']).all()
            assert ((obs['depth']>=0)&(obs['depth']<=1)).all()
            raw=camera.data.output['distance_to_camera'];raw_nonfinite+=int((~torch.isfinite(raw)).sum())
            depth_min=min(depth_min,float(obs['depth'].min()));depth_max=max(depth_max,float(obs['depth'].max()))
            depth_changed|=(obs['depth']!=previous_depth).reshape(n,-1).any(-1)
            previous_depth=obs['depth'].clone()
            camera_updates+=(camera.frame!=previous_frame).long();previous_frame=camera.frame.clone()
            action=model.act_inference(obs)
            assert action.shape==(n,8) and torch.isfinite(action).all()
            prior=robot.data.root_pos_w[:,0].clone()-start[:,0]
            obs,reward,dones,extras=env.step(action)
            active=~done;new=active&dones.bool()
            assert torch.isfinite(reward).all()
            contribution=base.reward_manager._step_reward*dt;assert torch.isfinite(contribution).all()
            errors=(contribution.double().sum(-1)-reward.double()).abs()[active]
            max_step_error=max(max_step_error,float(errors.max()));sum_step_error+=float(errors.sum());sample_count+=int(active.sum())
            returns[active]+=reward[active].double();official[active]+=reward[active];steps[active]+=1;components[active]+=contribution[active].double()
            velocity[active]+=torch.where(dones.bool(),terminal_v,robot.data.root_lin_vel_w[:,0])[active].double()
            current=robot.data.root_pos_w[:,0]-start[:,0]
            maximum[active&~dones.bool()]=torch.maximum(maximum[active&~dones.bool()],current[active&~dones.bool()])
            if new.any():
                end[new]=terminal_x[new];legacy[new]=prior[new];maximum[new]=torch.maximum(maximum[new],end[new])
                fall[new]=base.termination_manager.get_term(fallterm)[new]
                timeout[new]=base.termination_manager.get_term('time_out')[new]
                if a.environment=='teammate2':boundary[new]=base.termination_manager.get_term('map_boundary')[new]
            ids=active.nonzero().flatten().cpu().tolist();lengths=steps.cpu().tolist();rc=reward.cpu().tolist();cc=contribution.cpu().tolist();dc=dones.cpu().tolist();tc=base.reset_terminated.cpu().tolist();uc=base.reset_time_outs.cpu().tolist()
            for i in ids:w.writerow([i,lengths[i],lengths[i]*dt,rc[i]]+cc[i]+[bool(dc[i]),tc[i],uc[i]])
            done|=dones.bool()
        if (step+1)%120==0:print(f'[PROGRESS] {step+1} steps, {int(done.sum())}/{n} first episodes complete',flush=True)
        if done.all():break
    output.close();assert done.all(),'Not all first episodes completed'
    assert (camera_updates>0).all(),'Depth camera did not update'
    assert depth_changed.all(),'Depth images did not change during the rollout'
    residual=(components.sum(-1)-returns).abs()
    assert max_step_error<1e-5 and float(residual.max())<1e-3,'Reward identity residual exceeds tolerance'
    rows=[]
    for i in range(n):
        row=dict(env_id=i,episode_return=float(returns[i]),official_float32_episode_return=float(official[i]),episode_steps=int(steps[i]),episode_seconds=int(steps[i])*dt,
            forward_displacement=float(end[i]),legacy_preterminal_displacement=float(legacy[i]),maximum_forward_displacement=float(maximum[i]),mean_forward_velocity=float(velocity[i]/steps[i]),
            fall=bool(fall[i]),timeout=bool(timeout[i]),boundary_exit=bool(boundary[i]),reward_identity_residual=float(residual[i]),terrain_row=assignment['rows'][i],terrain_column=assignment['columns'][i])
        row.update({'reward_'+name:float(components[i,j]) for j,name in enumerate(TERMS)});rows.append(row)
    with (a.output/'reward_components_episode.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={k:stats([x[k] for x in rows]) for k in ('episode_return','episode_steps','episode_seconds','forward_displacement','mean_forward_velocity')}
    component_stats={name:stats(components[:,i].cpu().numpy()) for i,name in enumerate(TERMS)}
    with (a.output/'reward_components_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['component','mean','std','min','max','median']);writer.writeheader()
        for key,value in component_stats.items():writer.writerow(dict(component=key,**value))
        writer.writerow(dict(component='total',**summary['episode_return']))
    result=dict(environment=a.environment,task=TASKS[a.environment],original_task=ORIGINAL[a.environment],checkpoint=str(Path(a.checkpoint).resolve()),checkpoint_sha256=EXPECTED_SHA,saved_iteration=6632,
        seed=a.seed,terrain_seed=cfg.scene.terrain.terrain_generator.seed,num_envs=n,completed=int(done.sum()),first_episode_only=True,control_dt=dt,
        actor_input_dim=123,runtime_actor_features=123,runtime_proprio_dim=59,runtime_depth_shape=[48,64,1],depth_embedding_dim=64,action_dimension=8,
        host_config_equality=equal,observation_config_equals_training_source=True,depth_camera_config_equals_training_source=True,
        observation_terms=base.observation_manager.active_terms,observation_term_dimensions=base.observation_manager.group_obs_term_dim,
        runtime_validation=dict(observations_finite=True,normalized_depth_finite=True,normalized_depth_in_range=True,actions_finite=True,rewards_finite=True,
            depth_camera_updates_min=int(camera_updates.min()),normalized_depth_min=depth_min,normalized_depth_max=depth_max,depth_changed_all_envs=bool(depth_changed.all()),raw_depth_nonfinite_samples=raw_nonfinite,raw_depth_nonfinite_handling='Original native normalized_depth: replace NaN/Inf then clip0.1..5.0 and normalize',
            sensors=list(base.scene.sensors)),
        summary=summary,reward_components=component_stats,reward_definitions=definitions,
        fall_term=fallterm,fall_count=int(fall.sum()),fall_ratio=float(fall.float().mean()),timeout_count=int(timeout.sum()),timeout_ratio=float(timeout.float().mean()),
        other_termination_count=int(boundary.sum()),other_termination_ratio=float(boundary.float().mean()),other_termination='map_boundary' if a.environment=='teammate2' else None,
        reached_5m_count=int((maximum>=5).sum()),identity=dict(step_max_absolute_residual=max_step_error,step_mean_absolute_residual=sum_step_error/sample_count,
            episode_max_absolute_residual=float(residual.max()),episode_mean_absolute_residual=float(residual.mean()),float32_vs_float64_max_difference=float((official.double()-returns).abs().max()),all_finite=True),
        terrain_mesh_sha256=meshhash,env_origins_sha256=hashlib.sha256(origins.cpu().numpy().tobytes()).hexdigest(),terrain_assignment=assignment,
        initial_root_states=start.cpu().tolist(),robot_material_properties=robot.root_physx_view.get_material_properties().cpu().tolist(),joint_names=robot.joint_names,
        displacement_definition='Terminal-inclusive signed world-X displacement; read before original auto-reset. Native Final results use preterminal pose, which is also retained here.',
        accounting='active=~finished; accumulate env.step reward and1 step including terminal; finished|=dones; later reset episodes excluded',
        per_env=rows,command=sys.argv)
    dump(a.output/'evaluation_seed24_n100.json' if n==100 else a.output/'smoke.json',result)
    faulthandler.cancel_dump_traceback_later()
    env.close()
    print('[RESULT] '+json.dumps(dict(environment=a.environment,completed=n,summary=summary,fall=int(fall.sum()),timeout=int(timeout.sum()),other=int(boundary.sum()),identity=result['identity'])),flush=True)
try:main()
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:app.close()
