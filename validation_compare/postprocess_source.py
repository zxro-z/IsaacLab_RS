import json,csv,pathlib,hashlib,subprocess,shutil,math,shlex
from fractions import Fraction
ROOT=pathlib.Path('/home/zxro/teammate_ant_rl');RS=pathlib.Path('/home/zxro/IsaacLab_RS');T2=pathlib.Path('/home/zxro/teammate_ant_rl_2/validation_compare');D=ROOT/'validation_compare'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def write(p,text):
 with p.open('x') as f:f.write(text)
def dump(p,x):write(p,json.dumps(x,indent=2,allow_nan=False)+'\n')
m=load(D/'evaluation_seed24_n100/evaluation.json');s=load(D/'smoke_seed24/evaluation.json');v=load(D/'video_seed24_env0/evaluation.json');t2=load(T2/'evaluation_seed24_n100.json')
assert m['completed']==100 and s['completed']==4 and v['completed']==1
for src,dst in [('evaluation_seed24_n100/evaluation.json','evaluation_seed24_n100.json'),('evaluation_seed24_n100/reward_components_summary.csv','reward_components_seed24_n100.csv'),('evaluation_seed24_n100/reward_components_episode.csv','reward_components_episode_seed24_n100.csv'),('smoke_seed24/evaluation.json','smoke_seed24.json')]:
 assert not (D/dst).exists();shutil.copy2(D/src,D/dst)
for src,dst in [('/tmp/teammate1_smoke.log','smoke_seed24.log'),('/tmp/teammate1_n100.log','evaluation_seed24_n100.log'),('/tmp/teammate1_video.log','video_seed24_env0.log')]:
 assert not (D/dst).exists();shutil.copy2(src,D/dst)
mp4=list((D/'video_seed24_env0/video').glob('*.mp4'));assert len(mp4)==1
out=D/'contact_teammate1_seed24_env0.mp4';assert not out.exists();shutil.copy2(mp4[0],out)
probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames:format=duration,size','-of','json',str(out)],text=True));st=probe['streams'][0]
assert (st['width'],st['height'])==(1280,720) and Fraction(st['avg_frame_rate'])==60
subprocess.run(['ffmpeg','-v','error','-i',str(out),'-f','null','-'],check=True)
dump(D/'video_manifest.json',dict(output=str(out),sha256=sha(out),ffprobe=probe,full_decode_pass=True,task=v['task'],seed=24,terrain_seed=42,
 num_envs=1,env_index=0,episode_index=0,first_episode_only=True,checkpoint=v['checkpoint'],checkpoint_sha256=v['checkpoint_sha256'],dimensions=v['dimensions'],telemetry=v['per_env'][0],
 visualization=load(D/'video_seed24_env0/visualization.json'),terrain_mesh_sha256=v['terrain_mesh_sha256'],same_mesh_as_main=v['terrain_mesh_sha256']==m['terrain_mesh_sha256'],
 env_origin=v['terrain_assignment']['origins'][0],initial_root_state=v['initial_root_states'][0],
 note='Single-env video is not the 100-env env0 initial condition: vectorized tile assignment/reset/material sampling depends on num_envs. No episode selection.'))
old=load(pathlib.Path('/tmp/teammate1_team1_before.json'));unchanged={}
for name,digest in old['files'].items():
 p=ROOT/name;actual=sha(p) if p.is_file() else None;unchanged[name]=dict(before=digest,after=actual,unchanged=actual==digest)
assert all(x['unchanged'] for x in unchanged.values())
rsold=load(pathlib.Path('/tmp/teammate1_rs_before.json'))
now={str(f.relative_to(RS)):sha(f) for f in RS.rglob('*') if f.is_file() and not any(x in f.parts for x in ['.git','.agents','.codex','.aws','__pycache__'])}
assert now==rsold['files']
status=subprocess.check_output(['git','-C',str(RS),'status','--short'],text=True);head=subprocess.check_output(['git','-C',str(RS),'rev-parse','HEAD'],text=True).strip()
assert status==rsold['status'] and head==rsold['head']
integrity=dict(rs_files_checked=len(now),rs_all_files_unchanged=True,rs_status_before=rsold['status'],rs_status_after=status,rs_head_unchanged=True,
 teammate_original_files_checked=len(unchanged),teammate_original_files_unchanged=True,teammate_original_file_hashes=unchanged,
 checkpoint_sha256=m['checkpoint_sha256'],no_training=True,no_commit=True,no_push=True)
dump(D/'integrity.json',integrity)
config=D/'evaluation_seed24_n100/config.json';cfg=load(config)
source=dict(original_task='Ant-rl-v0',local_task=m['task'],teammate_git_commit=old['head'],original_registration='source/ant/__init__.py:18-26',
 original_environment='source/ant/ant_env_cfg.py:AntEnvCfg (298-323)',terrain='source/ant/ant_env_cfg.py:59-96',
 original_current_play_reward='scripts/rsl_rl/play.py:85-122 EvalRewardsCfg; assignment at152 before gym.make at155',
 original_training_reward='source/ant/ant_env_cfg.py:282 -> rewards.TotalReward; excluded',
 original_command='command.txt PLAY section: python scripts/rsl_rl/play.py --task Ant-rl-v0 --num_envs 4 --checkpoint logs/rsl_rl/ant/RUN_FOLDER/best_model.pt',
 original_command_status='Documented template, not a recovered historical executed command. Final reward output artifact, exact evaluation seed and num_envs were not found.',
 saved_configs={'v3':'logs/rsl_rl/ant/v3_depth/params/env.yaml','v2':'logs/rsl_rl/ant/v2/params/env.yaml'},
 environment_selection='Current registered Ant-rl-v0 plus current play.py and project.md; current terrain parameters match saved v3 config. v2 random-grid is a different historical config.',
 local_registration='scripts/evaluate_contact_teammate1.py:ContactEvaluationCfg and gym.register',
 host_geometry_asset_actions_events_terminations_sim_config_equality_passed=True,
 observation_change='Remove unused depth camera/group and original feet diagnostic sensor; add one RS 63-ray scanner and one RS four-foot sensor; stock60+63+4=127; no padding, slicing, or checkpoint conversion',
 reward_source=str(RS/'source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/ant_env_cfg.py')+':124-152',
 reward_manager_source=str(RS/'source/isaaclab/isaaclab/managers/reward_manager.py')+':149-156',
 main_source_hashes=load(D/'evaluation_seed24_n100/source_hashes.json'),config_sha256=sha(config),terrain_mesh_sha256=m['terrain_mesh_sha256'],env_origins_sha256=m['env_origins_sha256'],
 checkpoint=m['checkpoint'],checkpoint_sha256=m['checkpoint_sha256'],
 runtime={'isaac_sim_package':'5.0.0.0','isaac_sim_build':'5.0.0-rc.45+release.23960.184afb15.gl','isaaclab_source':str(RS),'rsl_rl':'3.0.1','torch':'2.7.0+cu128','gpu':'NVIDIA GeForce RTX 5070 Ti'},
 caveats=['Historical final reward execution provenance is incomplete; this is the verified current native evaluation environment, not a proven replay of an unknown historical final run.',
 'Teammate depth perception is replaced with checkpoint-required RS HeightScan/Contact. Geometry and physics definitions are retained; observation/camera differences are intentional.',
 'Host noninstanceable ant.usd is retained; RS checkpoint trained with stock instanceable Ant. Runtime joint order and effort action scale7.5 are saved.',
 'No terrain-relative height fall check: fall is body_z_down (>90deg). Different from FinalUnseen clearance<0.31m.',
 'Startup robot friction is random static[0.3,1.0], dynamic0.8*static; kept unchanged. Teammate2 uses different material/reset/map-boundary definitions.',
 'World-X displacement is terminal-inclusive here; original Final uses preterminal pose. The matching legacy metric is retained.',
 'No map-boundary termination is configured; travel can leave the initial 10m tile and cross other terrain tiles/border.',
 'Generator seed alone does not guarantee identical historical meshes across runtime/Torch RNG differences; per-run mesh/origin/material data is saved.'])
dump(D/'source_mapping.json',source)
commands=['# cwd=/home/zxro/teammate_ant_rl']
for label,result in [('smoke',s),('n100',m),('video',v)]:
 argv=result['command'];argv[0]='scripts/evaluate_contact_teammate1.py'
 commands.append('ISAAC_SIM_SITE_PACKAGES=/home/zxro/anaconda3/envs/env_isaaclab/lib/python3.11/site-packages PYTHONDONTWRITEBYTECODE=1 CONDA_PREFIX=/home/zxro/anaconda3/envs/env_isaaclab_231 /home/zxro/anaconda3/envs/env_isaaclab_231/bin/python '+shlex.join(argv)+' > /tmp/teammate1_'+label+'.log 2>&1')
commands+=['python /tmp/finalize_teammate1_report.py',
 'PYTHONPYCACHEPREFIX=/tmp/teammate1_pycache /home/zxro/anaconda3/envs/env_isaaclab_231/bin/python -m py_compile scripts/evaluate_contact_teammate1.py validation_compare/postprocess_source.py',
 'git -C /home/zxro/teammate_ant_rl diff --check',
 'ffprobe -v error -show_streams -show_format validation_compare/contact_teammate1_seed24_env0.mp4',
 'ffmpeg -v error -i validation_compare/contact_teammate1_seed24_env0.mp4 -f null -']
write(D/'commands.log','\n\n'.join(commands)+'\n')
write(D/'environment_provenance.md','''# Teammate 1 current evaluation environment

## Evidence and unresolved history
`source/ant/__init__.py:18-26` registers only Ant-rl-v0. `command.txt` PLAY section and `project.md` sections3,9,10 identify play.py + Ant-rl-v0. Current terrain and termination match saved `logs/rsl_rl/ant/v3_depth/params/env.yaml`; v2 saved config uses a different 8×8 random-grid terrain. No final reward-output log/result artifact was found. Exact historical final seed, num_envs and executed checkpoint command are therefore unknown. This measurement uses the confirmed current native evaluation host and does not claim to reproduce an unrecorded historical reward result.

Documented original command template (not executed here):
```bash
python scripts/rsl_rl/play.py --task Ant-rl-v0 --num_envs 4 --checkpoint logs/rsl_rl/ant/RUN_FOLDER/best_model.pt
```

## Geometry, physics and reset
- ROUGH_TERRAINS_CFG copy with native overrides (`source/ant/ant_env_cfg.py:59-96`), generator seed42, no curriculum, difficulty range[0,1].
- 20×10 tiles of10×10m; border2m; horizontal scale0.1m, vertical scale0.005m, slope threshold0.75, cache false.
- Each active subterrain proportion0.2: pyramid_stairs, pyramid_stairs_inv, boxes, hf_pyramid_slope, hf_pyramid_slope_inv. random_rough proportion0.
- Stairs height0.03–0.07m,width0.3m; boxes grid0.45m,height0.02–0.10m; slopes0–0.20; central platforms1m; native holes/border settings retained.
- Ground static/dynamic friction1, restitution0; both combine modes multiply.
- Robot static friction uniform[0.3,1.0] at startup, dynamic0.8×static, restitution0; unchanged across resets. Per-run material arrays saved.
- Stock host noninstanceable Ant asset, joint ordering and effort action scale7.5 retained.
- Root reset extra pose/velocity randomization empty; joint position offset[-0.2,0.2], velocity[-0.1,0.1]. Native importer terrain origins/assignment retained, no teammate2 reselect event.
- Physics dt1/120s, decimation2, control dt1/60s; 16s/960steps timeout.
- Fall only body_z_down, bad_orientation(limit_angle=pi/2); torso-height condition disabled; no boundary exit termination.

## Reward separation
Training RewardManager has one total_reward term whose custom function internally combines10 components. Native play.py overrides it with6 EvalRewardsCfg terms; it omits joint_pos_limits. Neither is used for Contact measurement. The local task explicitly instantiates RS stock RewardsCfg, unchanged weights/functions, with seven terms. Contributions are the existing manager._step_reward × control_dt (recovering its weighted buffer); no reward functions are recomputed. Env.step total is authoritative. Terminal step included, subsequent reset episodes excluded; float64 trace plus official float32 accumulation are recorded.

## Observation separation
The original59-D proprioception/depth group and depth camera are not used. The Contact checkpoint receives native RS60-D stock observation +63 RayCaster heights +4 binary foot contacts, validated127/127. Exactly one scanner and one feet sensor are present. Host asset/terrain/action/events/termination/sim config equality is asserted before launch.

## Limits
'''+'\n'.join('- '+x for x in source['caveats'])+'\n')
final=next(x for x in csv.DictReader((RS/'validation/final_unseen_results.csv').open()) if x['policy']=='HeightScan+Contact')
fs={x['component']:float(x['mean']) for x in csv.DictReader((RS/'validation/final_unseen_reward_components_summary.csv').open()) if x['policy']=='HeightScan+Contact' and x['measure']=='episode'}
lines=['# RS Contact in teammate1 current native evaluation environment','','Seed24, terrain seed42,100 first episodes. Population std (ddof=0). Teammate policy and reward modifications excluded. Current environment identity is source-confirmed; historical final reward execution log was unavailable.','',
 '## 100-env metrics','','| Metric | Value |','|---|---:|']
for name,d in m['summary'].items():lines.append(f"| {name} mean ± std | {d['mean']:.6f} ± {d['std']:.6f} |")
lines+=[f"| Return min/max | {m['summary']['episode_return']['min']:.6f} / {m['summary']['episode_return']['max']:.6f} |",f"| Fall / timeout | {m['fall_count']}/100 / {m['timeout_count']}/100 |",f"| Observed world-X ≥5m | {m['reached_5m_count']}/100 |",'',
 '## Stock weighted contributions','','Each term is function×weight×control_dt. `energy` is a stock reward proxy, not literal physical energy.','',
 '| Component | Teammate1 mean ± std | FinalUnseen mean | Teammate2 mean |','|---|---:|---:|---:|']
for term,d in m['reward_components'].items():lines.append(f"| {term} | {d['mean']:.6f} ± {d['std']:.6f} | {fs[term]:.6f} | {t2['reward_components'][term]['mean']:.6f} |")
lines+=[f"| total env.step | {m['summary']['episode_return']['mean']:.6f} ± {m['summary']['episode_return']['std']:.6f} | 69.610059 | {t2['summary']['episode_return']['mean']:.6f} |",'',
 '## Reward identity','',f"Step max residual={m['identity']['max_absolute_step_residual']:.9g}; episode max/mean={m['identity']['max_absolute_episode_residual']:.9g}/{m['identity']['mean_absolute_episode_residual']:.9g}.",
 f"Maximum official float32 vs float64 accumulated difference={m['identity']['max_float32_accumulation_difference']:.9g}. All finite; no reward function/weight/manager changes.",'',
 '## Same policy across three environments','','| Environment | Return mean ± std | Displacement mean ± std (m) | Fall | Timeout | Other termination |','|---|---:|---:|---:|---:|---|',
 '| My FinalUnseen | 69.610 ± 16.748 | 61.914 ± 14.977 (preterminal) |16%|84%|—|',
 f"| Teammate1 current host | {m['summary']['episode_return']['mean']:.3f} ± {m['summary']['episode_return']['std']:.3f} | {m['summary']['forward_displacement']['mean']:.3f} ± {m['summary']['forward_displacement']['std']:.3f} (terminal inclusive) | {m['fall_count']}% | {m['timeout_count']}% | none configured |",
 f"| Teammate2 SelfEval | {t2['summary']['episode_return']['mean']:.3f} ± {t2['summary']['episode_return']['std']:.3f} | {t2['summary']['forward_displacement']['mean']:.3f} ± {t2['summary']['forward_displacement']['std']:.3f} (terminal inclusive) | {t2['fall_count']}% | {t2['timeout_count']}% | boundary18% |",'',
 f"Matching preterminal displacement for teammate1: {sum(x['legacy_preterminal_displacement'] for x in m['per_env'])/100:.6f}m. Terminal-inclusive telemetry avoids auto-reset teleportation.",
 'Final uses clearance-based fall; teammates use >90° orientation. Teammate2 has boundary truncation and a different spawn/terrain-reset rule. Teammate1 preserves startup friction randomization and has no map-boundary termination. Percentages therefore do not define an overall cross-environment robustness ranking.',
 'In this host,32/100 runs ended by orientation fall and shorter episodes reduced alive accumulation. Accumulated progress is below the frozen Final value. Smaller accumulated energy proxy penalty than teammate2 cannot alone establish more efficient control because durations differ. Different terrain realizations and physics/reset conditions preclude a causal attribution.','',
 '## Video','',f"[Seed24 env0 first episode](contact_teammate1_seed24_env0.mp4): {probe['format']['duration']}s,1280×720,60fps.",
 f"Single-env telemetry: return={v['per_env'][0]['episode_return']:.6f}, signed-X displacement={v['per_env'][0]['forward_displacement']:.6f}m, steps={v['per_env'][0]['episode_steps']}, fall={v['per_env'][0]['fall']}, timeout={v['per_env'][0]['timeout']}.",
 'Qualitative only, first episode without selection. Native vectorized randomization means n1 env0 is not n100 env0. Camera/material/light edits are visual only and physical mesh/material signatures are checked unchanged.',
 '', '[Provenance](environment_provenance.md) · [Source mapping](source_mapping.json) · [Commands](commands.log) · [Integrity](integrity.json) · [Video manifest](video_manifest.json)']
write(D/'reward_decomposition_summary.md','\n'.join(lines)+'\n')
for p in D.rglob('*.json'):load(p)
for p in D.rglob('*.csv'):
 for row in csv.reader(p.open()):assert not any(x.lower() in ('nan','inf','-inf','infinity') for x in row)
rows=list(csv.DictReader((D/'reward_components_episode_seed24_n100.csv').open()));assert len(rows)==100
assert sorted(int(x['env_id']) for x in rows)==list(range(100))
assert sum(x['episode_steps'] for x in m['per_env'])==sum(1 for _ in csv.DictReader((D/'evaluation_seed24_n100/reward_components_step.csv').open()))
print(json.dumps(dict(video_probe=probe,video_telemetry=v['per_env'][0],video_mesh_equal_main=v['terrain_mesh_sha256']==m['terrain_mesh_sha256'],rs_unchanged=True,original_team_files_unchanged=len(unchanged)),indent=2))
