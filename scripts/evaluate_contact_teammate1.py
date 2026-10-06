"""Evaluation-only Contact adapter; original teammate files and RS checkout are read-only."""
import os, sys, argparse, json, hashlib, csv, subprocess, types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RS=Path('/home/zxro/IsaacLab_RS')
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'source'))
if os.environ.get('ISAAC_SIM_SITE_PACKAGES'): sys.path.append(os.environ['ISAAC_SIM_SITE_PACKAGES'])
for package in ('isaaclab','isaaclab_assets','isaaclab_tasks','isaaclab_rl'):
    sys.path.insert(0,str(RS/'source'/package))
sys.path.insert(0,str(RS/'scripts/reinforcement_learning/rsl_rl'))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--task',default='Isaac-Ant-Teammate1-RewardEnv-v0')
p.add_argument('--checkpoint',required=True)
p.add_argument('--seed',type=int,default=24)
p.add_argument('--num_envs',type=int,default=100)
p.add_argument('--height_scan',action='store_true',required=True)
p.add_argument('--foot_contacts',action='store_true',required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--video',action='store_true')
AppLauncher.add_app_launcher_args(p)
a=p.parse_args()
if a.output.exists(): raise FileExistsError(a.output)
a.output.mkdir(parents=True)
if a.video: a.enable_cameras=True
app=AppLauncher(a).app
import gymnasium as gym
import torch, numpy as np
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import RewardsCfg, ObservationsCfg
from isaaclab_tasks.manager_based.classic.ant.agents.rsl_rl_ppo_cfg import AntPPORunnerCfg
from isaaclab_tasks.manager_based.classic.ant.ant_contact_observations import add_foot_contacts_for_evaluation
from ant_observation_adapter import configure_height_scan,validate_checkpoint_inputs,scan_metadata
from ant.ant_env_cfg import AntEnvCfg as TeammateHostCfg
from isaaclab.utils import configclass
from pxr import UsdGeom

@configclass
class ContactEvaluationCfg(TeammateHostCfg):
    """Keep original native/current Ant-rl-v0 scene/physics/events/terminations, replace only policy inputs and reward."""
    def __post_init__(self):
        super().__post_init__()
        self.rewards=RewardsCfg()
        self.observations=ObservationsCfg()
        # One feet sensor and one scanner: replace original perception without padding or slicing.
        self.scene.contact_forces=None
        self.scene.depth_camera=None
        configure_height_scan(self,enabled=True)
        add_foot_contacts_for_evaluation(self)

gym.register(id=a.task,entry_point='isaaclab.envs:ManagerBasedRLEnv',disable_env_checker=True,
    kwargs={'env_cfg_entry_point':ContactEvaluationCfg,
            'rsl_rl_cfg_entry_point':'isaaclab_tasks.manager_based.classic.ant.agents.rsl_rl_ppo_cfg:AntPPORunnerCfg'})

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value): Path(path).write_text(json.dumps(value,indent=2,allow_nan=False,default=str)+'\n')
def stats(x):
    x=np.asarray(x,dtype=np.float64)
    return dict(mean=float(x.mean()),std=float(x.std(ddof=0)),min=float(x.min()),max=float(x.max()),median=float(np.median(x)))
def main():
    cfg=ContactEvaluationCfg();cfg.seed=a.seed;cfg.scene.num_envs=a.num_envs
    agent=AntPPORunnerCfg();agent.seed=a.seed
    if a.device: cfg.sim.device=agent.device=a.device
    cfg.log_dir=str(a.output)
    cfg.viewer.resolution=(1280,720)
    if a.video:
        cfg.viewer.origin_type='asset_root';cfg.viewer.asset_name='robot';cfg.viewer.env_index=0
    host=TeammateHostCfg()
    for key in ('terrain','robot'):
        original=getattr(host.scene,key).to_dict();adapted=getattr(cfg.scene,key).to_dict()
        assert original==adapted, f'Host {key} changed'
    for key in ('actions','events','terminations','sim'):
        assert getattr(host,key).to_dict()==getattr(cfg,key).to_dict(),f'Host {key} changed'
    dump(a.output/'config.json',cfg.to_dict())
    sources=[ROOT/'source/ant/ant_env_cfg.py',ROOT/'scripts/rsl_rl/play.py',ROOT/'source/ant/__init__.py',ROOT/'logs/rsl_rl/ant/v3_depth/params/env.yaml',
             RS/'source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/ant_env_cfg.py',
             RS/'source/isaaclab/isaaclab/managers/reward_manager.py',Path(__file__)]
    dump(a.output/'source_hashes.json',{str(x):sha(x) for x in sources})
    env=gym.make(a.task,cfg=cfg,render_mode='rgb_array' if a.video else None)
    base=env.unwrapped
    visual=None
    if a.video:
        from rollout_visualization import TerrainReliefVisualization,FirstEpisodeReliefVideo
        visual=TerrainReliefVisualization(base)
        env=FirstEpisodeReliefVideo(env,video_folder=str(a.output/'video'),step_trigger=lambda step:step==0,
            video_length=base.max_episode_length,disable_logger=True,name_prefix='contact_teammate1_seed24_env0')
        env.metadata['render_fps']=60
    env=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions)
    dimensions=validate_checkpoint_inputs(a.checkpoint,base)
    assert dimensions['actor_input_dim']==127 and env.num_actions==8
    runner=OnPolicyRunner(env,agent.to_dict(),log_dir=None,device=agent.device)
    runner.load(a.checkpoint)
    policy=runner.get_inference_policy(device=base.device)
    obs=env.get_observations()
    robot=base.scene['robot'];n=env.num_envs;dt=base.step_dt
    start=robot.data.root_state_w.clone()
    initial_origins=base.scene.env_origins.clone()
    assignment={'rows':base.scene.terrain.terrain_levels.cpu().tolist(),
        'columns':base.scene.terrain.terrain_types.cpu().tolist(),'origins':initial_origins.cpu().tolist()}
    mesh=UsdGeom.Mesh(base.scene.stage.GetPrimAtPath('/World/ground/terrain/mesh'))
    meshhash=hashlib.sha256(np.asarray(mesh.GetPointsAttr().Get(),dtype=np.float32).tobytes()+np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),dtype=np.int32).tobytes()).hexdigest()
    terms=list(base.reward_manager.active_terms)
    assert terms==['progress','alive','upright','move_to_target','action_l2','energy','joint_pos_limits']
    definitions=[]
    for name in terms:
        term=base.reward_manager.get_term_cfg(name);func=term.func;owner=func if hasattr(func,'__qualname__') else type(func)
        definitions.append(dict(name=name,weight=term.weight,function=owner.__module__+'.'+owner.__qualname__,params=term.params))
    finished=torch.zeros(n,dtype=torch.bool,device=base.device)
    returns=torch.zeros(n,dtype=torch.float64,device=base.device)
    official_returns=torch.zeros(n,device=base.device)
    lengths=torch.zeros(n,dtype=torch.long,device=base.device)
    sums=torch.zeros((n,7),dtype=torch.float64,device=base.device)
    velocity=torch.zeros(n,dtype=torch.float64,device=base.device)
    endpoint=torch.zeros(n,device=base.device);maxdis=torch.zeros(n,device=base.device)
    failures=torch.zeros(n,dtype=torch.bool,device=base.device);timeouts=failures.clone();boundary=failures.clone()
    terminal_velocity=torch.zeros(n,device=base.device)
    terminal_x=torch.zeros(n,device=base.device)
    legacy_displacement=torch.zeros(n,device=base.device)
    reset_method=base._reset_idx
    def read_terminal(self,ids):
        terminal_velocity[ids]=robot.data.root_lin_vel_w[ids,0]
        terminal_x[ids]=robot.data.root_pos_w[ids,0]-start[ids,0]
        return reset_method(ids)
    base._reset_idx=types.MethodType(read_terminal,base)
    max_step_error=0.;abs_step_error=0.;step_samples=0
    checks={'observations_finite':True,'height_scan_finite':True,'contacts_finite_binary':True,'actions_finite':True,'rewards_finite':True}
    stepfile=(a.output/'reward_components_step.csv').open('w',newline='')
    writer=csv.writer(stepfile);writer.writerow(['env_id','step','time_s','total_reward']+['reward_'+t for t in terms]+['done','terminated','truncated'])
    for step in range(base.max_episode_length):
        assert app.is_running(),'Simulator closed before complete batch'
        if visual:visual.update_camera()
        with torch.inference_mode():
            ob=obs['policy'];assert ob.shape==(n,127)
            assert torch.isfinite(ob).all(),'Nonfinite observations'
            assert torch.isfinite(ob[:,60:123]).all()
            assert ((ob[:,-4:]==0)|(ob[:,-4:]==1)).all(),'Contacts not binary'
            actions=policy(obs);assert actions.shape==(n,8) and torch.isfinite(actions).all()
            prior=robot.data.root_pos_w[:,0].clone()-start[:,0]
            obs,reward,done,extra=env.step(actions)
            active=~finished;new=active&done.bool()
            assert torch.isfinite(reward).all()
            components=base.reward_manager._step_reward*dt
            assert torch.isfinite(components).all()
            difference=(components.double().sum(-1)-reward.double()).abs()[active]
            max_step_error=max(max_step_error,float(difference.max()));abs_step_error+=float(difference.sum());step_samples+=int(active.sum())
            returns[active]+=reward[active].double();official_returns[active]+=reward[active]
            sums[active]+=components[active].double();lengths[active]+=1
            current=robot.data.root_pos_w[:,0]-start[:,0]
            velocity[active]+=torch.where(done.bool(),terminal_velocity,robot.data.root_lin_vel_w[:,0])[active].double()
            maxdis[active&~done.bool()]=torch.maximum(maxdis[active&~done.bool()],current[active&~done.bool()])
            if new.any():
                endpoint[new]=terminal_x[new]
                legacy_displacement[new]=prior[new]
                maxdis[new]=torch.maximum(maxdis[new],endpoint[new])
                failures[new]=base.termination_manager.get_term('body_z_down')[new]
                timeouts[new]=base.termination_manager.get_term('time_out')[new]
                boundary[new]=False
            ids=active.nonzero().flatten().cpu().tolist();rc=reward.cpu().tolist();cc=components.cpu().tolist();dc=done.cpu().tolist()
            terminated=base.reset_terminated.cpu().tolist();truncated=base.reset_time_outs.cpu().tolist()
            for i in ids:writer.writerow([i,int(lengths[i]),float(lengths[i])*dt,rc[i]]+cc[i]+[bool(dc[i]),terminated[i],truncated[i]])
            finished|=done.bool()
        if finished.all():break
    stepfile.close();assert finished.all(),'Incomplete first episodes'
    residual=(sums.sum(-1)-returns).abs()
    assert max_step_error<1e-5 and float(residual.max())<1e-3,'Reward identity failed'
    rows=[]
    for i in range(n):
        row=dict(env_id=i,episode_return=float(returns[i]),official_float32_episode_return=float(official_returns[i]),episode_steps=int(lengths[i]),
            episode_seconds=float(lengths[i])*dt,forward_displacement=float(endpoint[i]),legacy_preterminal_displacement=float(legacy_displacement[i]),
            maximum_forward_displacement=float(maxdis[i]),mean_forward_velocity=float(velocity[i]/lengths[i]),
            fall=bool(failures[i]),timeout=bool(timeouts[i]),map_boundary_exit=bool(boundary[i]),
            reward_identity_residual=float(residual[i]),terrain_row=assignment['rows'][i],terrain_column=assignment['columns'][i])
        row.update({'reward_'+t:float(sums[i,j]) for j,t in enumerate(terms)});rows.append(row)
    with (a.output/'reward_components_episode.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={name:stats([r[name] for r in rows]) for name in ('episode_return','episode_steps','episode_seconds','forward_displacement','mean_forward_velocity')}
    contributions={t:stats(sums[:,j].cpu().numpy()) for j,t in enumerate(terms)}
    with (a.output/'reward_components_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['component','mean','std','min','max','median']);w.writeheader()
        for name,data in contributions.items():w.writerow({'component':name,**data})
        w.writerow({'component':'total',**summary['episode_return']})
    result=dict(task=a.task,original_task='Ant-rl-v0',num_envs=n,completed=int(finished.sum()),seed=a.seed,
        terrain_seed=cfg.scene.terrain.terrain_generator.seed,control_dt=dt,episode_limit_seconds=cfg.episode_length_s,
        checkpoint=a.checkpoint,checkpoint_sha256=sha(a.checkpoint),dimensions=dimensions,action_dimension=8,
        joint_names=robot.joint_names,action_joint_names=base.action_manager.get_term('joint_effort')._joint_names,
        action_scale=7.5,scan=scan_metadata(base),runtime_validation=checks,summary=summary,
        fall_count=int(failures.sum()),fall_ratio=float(failures.float().mean()),timeout_count=int(timeouts.sum()),timeout_ratio=float(timeouts.float().mean()),
        boundary_exit_count=int(boundary.sum()),boundary_exit_ratio=float(boundary.float().mean()),
        reached_5m_count=int((maxdis>=5).sum()),reached_40m_count=int((maxdis>=40).sum()),
        reward_definitions=definitions,reward_components=contributions,
        identity=dict(max_absolute_step_residual=max_step_error,mean_absolute_step_residual=abs_step_error/step_samples,
            max_absolute_episode_residual=float(residual.max()),mean_absolute_episode_residual=float(residual.mean()),
            max_float32_accumulation_difference=float((official_returns.double()-returns).abs().max()),all_finite=True),
        terrain_mesh_sha256=meshhash,env_origins_sha256=hashlib.sha256(initial_origins.cpu().numpy().tobytes()).hexdigest(),
        terrain_assignment=assignment,initial_root_states=start.cpu().tolist(),
        robot_material_properties=robot.root_physx_view.get_material_properties().cpu().tolist(),
        accounting='active = ~finished; include env.step terminal reward and step; finished |= done; post-reset excluded',
        displacement_definition='Terminal-inclusive signed world-X displacement, read before original host auto-reset. Legacy preterminal values also retained.',
        termination_definition='fall=body_z_down (>pi/2), timeout=time_out; no map-boundary termination configured',
        per_env=rows,command=sys.argv,versions={'torch':torch.__version__,'isaaclab_source_root':str(RS)})
    dump(a.output/'evaluation.json',result)
    if visual:
        after=visual.physical_signature()
        # Host reset selects a new tile: origin changes are reset behavior, not a visualization modification.
        for key in visual.before:
            if key != 'env_origins_sha256': assert visual.before[key]==after[key],key
        dump(a.output/'visualization.json',dict(camera_eye_heading_frame=visual.EYE,lookat_heading_frame=visual.LOOKAT,
            camera_logic='Yaw-only rear-oblique asset-root follow; decision-time quaternion',
            terrain_visual_diffuse=[.48,.50,.52],roughness=.9,metallic=0,
            distant_light=dict(intensity=3000,rotation_xyz_deg=[0,-78,-35],color=[1,.96,.90],angle=.5),
            dome_fill=dict(intensity=120,color=[.85,.90,1]),
            physical_signature_before=visual.before,physical_signature_after=after,
            physics_geometry_unchanged=True,physics_material_unchanged=True,
            origin_note='Original host resets at its fixed assigned terrain origin; behavior retained',
            terminal_frame='Hold preterminal image for one frame; no reset pose or extra episode'))
    env.close()
    print('[RESULT] '+json.dumps({k:result[k] for k in ('completed','summary','fall_count','timeout_count','boundary_exit_count','identity')},allow_nan=False),flush=True)
try:main()
finally:app.close()
