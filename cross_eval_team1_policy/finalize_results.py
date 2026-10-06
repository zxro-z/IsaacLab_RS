import json,csv,math,hashlib,subprocess,py_compile
from pathlib import Path
root=Path('/home/zxro/teammate_ant_rl');out=root/'cross_eval_team1_policy'
terms=['progress','alive','upright','move_to_target','action_l2','energy','joint_pos_limits']
hosts=['final_unseen','teammate1','teammate2'];labels=['FinalUnseen','Teammate 1','Teammate 2']
def writej(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(r,*args):return subprocess.check_output(['git','-C',str(r),*args],text=True)
results={h:json.loads((out/h/'main/evaluation_seed24_n100.json').read_text()) for h in hosts}
validation={}
for h,d in results.items():
 assert d['num_envs']==d['completed']==100
 assert d['checkpoint_sha256']=='29a9474cc6e6d488b50ca33b9bc7ec99b516fdfc4313c3663d3843535990c2c5'
 assert d['identity']['step_max_absolute_residual']<1e-5
 assert d['identity']['episode_max_absolute_residual']<1e-3
 assert d['identity']['official_float32_component_sum_max_difference']<1e-3
 assert all(d['host_config_equality'].values())
 assert d['runtime_validation']['depth_changed_all_envs']
 paths=list((out/h/'main').glob('*.csv'));rows={}
 for p in paths:
  with p.open() as f:
   reader=csv.DictReader(f);count=0
   for r in reader:
    count+=1
    for v in r.values():
     try:x=float(v)
     except (ValueError,TypeError):continue
     assert math.isfinite(x),(p,v)
   rows[p.name]=count
 assert rows['reward_components_episode.csv']==100
 validation[h]={'csv_rows':rows,'identity':d['identity'],'episodes':d['completed']}
with (out/'comparison.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['component']+hosts)
 for t in terms:w.writerow([t]+[results[h]['reward_components'][t]['mean'] for h in hosts])
 w.writerow(['total']+[results[h]['summary']['episode_return']['mean'] for h in hosts])
with (out/'main_metrics.csv').open('w',newline='') as f:
 fields=['environment','return_mean','return_std','return_min','return_max','episode_steps_mean','episode_steps_std','episode_seconds_mean','displacement_mean','displacement_std','mean_forward_velocity','fall_count','fall_ratio','timeout_count','timeout_ratio','other_count','other_ratio','reach_5m_count']
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for h,d in results.items():
  s=d['summary'];w.writerow(dict(environment=h,return_mean=s['episode_return']['mean'],return_std=s['episode_return']['std'],return_min=s['episode_return']['min'],return_max=s['episode_return']['max'],episode_steps_mean=s['episode_steps']['mean'],episode_steps_std=s['episode_steps']['std'],episode_seconds_mean=s['episode_seconds']['mean'],displacement_mean=s['forward_displacement']['mean'],displacement_std=s['forward_displacement']['std'],mean_forward_velocity=s['mean_forward_velocity']['mean'],fall_count=d['fall_count'],fall_ratio=d['fall_ratio'],timeout_count=d['timeout_count'],timeout_ratio=d['timeout_ratio'],other_count=d['other_termination_count'],other_ratio=d['other_termination_ratio'],reach_5m_count=d['reached_5m_count']))
manifest=json.loads((out/'policy_manifest.json').read_text())
sources={h:json.loads((out/h/'main/source_hashes.json').read_text()) for h in hosts}
writej(out/'source_mapping.json',{'hosts':{h:{'original_task':results[h]['original_task'],'local_task':results[h]['task'],'source_hashes':sources[h],'host_config_equality':results[h]['host_config_equality'],'camera':str(root/'source/ant/ant_env_cfg.py')+':127','observation':str(root/'source/ant/ant_env_cfg.py')+':175','depth_preprocessing':str(root/'source/ant/depth_obs.py')+':13','native_depth_network':str(root/'source/ant/depth_actor_critic.py')+':41'} for h in hosts}})
lines=['# Team 1 v3 depth policy — three-environment stock-reward evaluation','',manifest['caveat'],'','Selected checkpoint: `logs/rsl_rl/ant/v3_depth/best_model.pt`; iteration 6632; SHA256 `'+manifest['sha256']+'`. Selection was fixed before evaluation and unchanged.','', '## Protocol','', 'Environment seed 24, 100 parallel environments per host, deterministic `act_inference`, first episode only. Terminal reward and terminal step are included; later reset episodes are excluded. Population standard deviation (`ddof=0`). Native host geometry, asset, physics, action mapping, initialization and termination are unchanged. The original 59-D proprioception and 48×64×1 depth observation, native normalization, 15 Hz camera and original depth-CNN are used on every host. No Contact 127-D adapter is used.','', 'Primary total return uses official float32 accumulation; float64 trace and per-component accumulation are retained separately. Weighted components are read from RewardManager (`_step_reward × control_dt`); reward functions are not recomputed. Contributions are `raw × weight × dt`.','', '## Mean accumulated weighted contributions','', '| Component | FinalUnseen | Teammate 1 | Teammate 2 |','|---|---:|---:|---:|']
for t in terms:lines.append('| '+t+' | '+' | '.join(f"{results[h]['reward_components'][t]['mean']:.3f}" for h in hosts)+' |')
lines.append('| **total** | '+' | '.join(f"**{results[h]['summary']['episode_return']['mean']:.3f}**" for h in hosts)+' |')
lines+=['','## Main metrics','','| Environment | Return mean ± std | Displacement mean ± std (m) | Fall | Timeout | Other |','|---|---:|---:|---:|---:|---:|']
for h,label in zip(hosts,labels):
 d=results[h];s=d['summary'];lines.append(f"| {label} | {s['episode_return']['mean']:.3f} ± {s['episode_return']['std']:.3f} | {s['forward_displacement']['mean']:.3f} ± {s['forward_displacement']['std']:.3f} | {d['fall_count']}% | {d['timeout_count']}% | {d['other_termination_count']}% |")
lines+=['','| Environment | Steps mean ± std | Mean duration (s) | Mean forward velocity (m/s) | Return min / max | ≥5m reach count |','|---|---:|---:|---:|---:|---:|']
for h,label in zip(hosts,labels):
 d=results[h];s=d['summary'];lines.append(f"| {label} | {s['episode_steps']['mean']:.2f} ± {s['episode_steps']['std']:.2f} | {s['episode_seconds']['mean']:.3f} | {s['mean_forward_velocity']['mean']:.3f} | {s['episode_return']['min']:.3f} / {s['episode_return']['max']:.3f} | {d['reached_5m_count']} |")
lines+=['','## Reward identity','','| Host | Step max residual | Episode max residual (float64 trace) | Episode mean residual | Float32 vs float64 max |','|---|---:|---:|---:|---:|']
for h,label in zip(hosts,labels):
 q=results[h]['identity'];lines.append(f"| {label} | {q['step_max_absolute_residual']:.3e} | {q['episode_max_absolute_residual']:.3e} | {q['episode_mean_absolute_residual']:.3e} | {q['float32_vs_float64_max_difference']:.3e} |")
lines+=['','## Smoke and runtime checks','', 'All three hosts passed n=4 smoke before their n=100 evaluation. Runtime proprioception59 + depth48×64×1 → CNN64 → actor123; action8. Normalized inputs/actions/rewards and components are finite. Depth values change during rollout, and camera updates were observed in every environment. Sky/background handling is the original `normalized_depth` implementation.','', 'Two initial teammate1 diagnostics stopped before policy rollout because the new script initially referenced `base.num_actions` instead of the wrapper’s `env.num_actions`; error shutdown was delayed by renderer teardown. This evaluation-only assertion was corrected. No host/camera/reward/policy settings were changed. The failed initialization attempts and successful smoke are preserved separately.','', '## Caveats','', '- Teammate 1 is the current native training-distribution host, not an independent unseen holdout. FinalUnseen and Teammate 2 are additional evaluation hosts.','- Fall differs by host: FinalUnseen uses terrain-relative torso clearance <0.31 m (including missing terrain hit); Teammate 1/2 use orientation termination. Timeout is 16 s. Teammate 2 additionally truncates at map boundary; it is reported as Other rather than timeout. Termination counts are native term outcomes.','- Terrain seeds remain 2404 / 42 / 71. Different terrain/reset/material/asset configurations and termination definitions prevent treating cross-environment fall rates or returns as a robustness ranking.','- Signed world-X displacement includes terminal position captured immediately before original auto-reset. The legacy preterminal displacement is retained in per-episode CSV. This differs slightly from preterminal displacement in older native Final reports. Mean forward velocity is the mean of each episode’s stepwise world-X velocity average.','- `energy` is the stock action/joint-velocity proxy, not literal measured physical energy.','- Original depth camera optics/extrinsics/preprocessing/timing are preserved across hosts; added sensing supplies the required policy interface and does not alter collision geometry. Simulator rendering/runtime version may differ from original training. Current runtime uses IsaacSim5.0.0-rc.45 and RSL-RL3.0.1 with the RS IsaacLab source checkout; exact original runtime equivalence is not asserted.','- Evaluation uses one seed and one realization per host. No post-evaluation checkpoint selection, training, video, commit or push was performed.','', 'Full precision: [reward table](comparison.csv), [main metrics](main_metrics.csv). Per-host `main/` contains config, source hashes, first-episode CSVs, step contributions and evaluation JSON. `commands.log` records execution commands.']
(out/'comparison.md').write_text('\n'.join(lines)+'\n')
protection={}
for k in ['rs','team1','team2']:
 b=json.loads(Path('/tmp/team1_depth_'+k+'_before.json').read_text());r=Path(b['root'])
 bad=[f for f,v in b['files'].items() if not (r/f).is_file() or sha(r/f)!=v]
 assert not bad,(k,bad)
 current=git(r,'status','--short');head=git(r,'rev-parse','HEAD').strip();assert head==b['head']
 if k!='team1':assert current==b['status'],(k,current,b['status'])
 protection[k]={'root':str(r),'original_file_count':len(b['files']),'modified_or_missing':bad,'head_before':b['head'],'head_after':head,'status_before':b['status'],'status_after':current,'before_snapshot_sha256':sha(Path('/tmp/team1_depth_'+k+'_before.json'))}
known=json.loads(Path('/tmp/team1_existing_untracked_before.json').read_text());assert len(known)==37
assert all(sha(root/p)==v for p,v in known.items())
tracked_diff=git(root,'diff','--name-only','origin/main');assert not tracked_diff
assert git(root,'rev-parse','HEAD')==git(root,'rev-parse','origin/main')
for r in [root,Path('/home/zxro/IsaacLab_RS'),Path('/home/zxro/teammate_ant_rl_2')]:subprocess.run(['git','-C',str(r),'diff','--check'],check=True)
for p in out.glob('*.py'):py_compile.compile(str(p),cfile='/tmp/team1_final_'+p.name+'.pyc',doraise=True)
for p in out.rglob('*.json'):json.loads(p.read_text())
writej(out/'integrity.json',{'protection':protection,'known_untracked_files':37,'known_untracked_preserved':True,'tracked_source_parity':'PASS','tracked_diff':0,'ahead_behind':git(root,'rev-list','--left-right','--count','HEAD...origin/main').strip(),'working_tree_clean':False,'checkpoint_sha256_after':sha(Path(manifest['checkpoint'])),'validation':validation,'json_parse':'PASS','csv_finite':'PASS','py_compile':'PASS','git_diff_check':'PASS','training':False,'commit':False,'push':False})
print('\n'.join(lines[lines.index('## Mean accumulated weighted contributions'):lines.index('## Smoke and runtime checks')]))
print('INTEGRITY PASS; original files and37 artifacts unchanged')
