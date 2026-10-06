"""Recover metadata after completed Baseline training's post-training path error.

No simulation, training, checkpoint writing or selection by evaluation occurs.
Preserves the original exit-1 log and records exactly what was verified/recovered.
"""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import torch

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/observation_ablation/baseline_4096x32x1000'
RUN=ROOT/'logs/rsl_rl/observation_ablation/ablation_baseline_stock_s42_e4096_n32_i1000'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name,value):
    path=BASE/name;assert not path.exists(),path
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


with (RUN/'training_curve.csv').open() as stream: curve=list(csv.DictReader(stream))
assert len(curve)==1000 and [int(r['iteration']) for r in curve]==list(range(1000))
best=max((r for r in curve if r['mean_reward']),key=lambda r:float(r['mean_reward']))
selected=RUN/'best_model.pt';final=RUN/'model_999.pt'
saved=torch.load(selected,map_location='cpu',weights_only=False)
last=torch.load(final,map_location='cpu',weights_only=False)
assert last['iter']==999 and saved['iter']==int(best['iteration'])
assert saved['infos']['mean_reward']==float(best['mean_reward'])
assert saved['model_state_dict']['actor.0.weight'].shape[1]==59
assert all(torch.isfinite(t).all() for t in saved['model_state_dict'].values())
assert sha(selected)==sha(BASE/'checkpoints/best_model.pt') and sha(final)==sha(BASE/'checkpoints/final_model.pt')
log=(BASE/'train_stdout.log').read_text()
assert 'Total timesteps: 131072000' in log and 'Learning iteration 999/1000' in log
assert "is not in the subpath" in log and "checkpoint_selection.json" in log
elapsed=re.findall(r'Time elapsed: (\d+):(\d+):(\d+)',log)[-1]
elapsed_seconds=sum(int(v)*scale for v,scale in zip(elapsed,(3600,60,1)))
manifest=json.loads((RUN/'manifest.json').read_text())
cfg=json.loads((RUN/'config.json').read_text());agent=json.loads((RUN/'agent.json').read_text())
assert cfg['scene']['num_envs']==4096 and cfg['seed']==agent['seed']==42
assert agent['num_steps_per_env']==32 and agent['max_iterations']==1000 and not agent['resume']
rule=manifest['protocol']['checkpoint_selection']
save('checkpoint_selection.json',dict(rule=rule,selected_iteration=saved['iter'],selected_training_mean_reward=saved['infos']['mean_reward'],
     curve=str((RUN/'training_curve.csv').relative_to(ROOT)),evaluation_not_used=True,checkpoint_sha256=sha(selected)))
end=datetime.fromtimestamp(final.stat().st_mtime,timezone.utc).isoformat()
save('training_summary.json',dict(completed=True,iterations=1000,num_steps_per_env=32,total_transitions=131072000,actual_total_transitions=131072000,
     actual_completed_iterations=1000,resume=False,training_seed=42,terrain_seed=42,launch_timestamp=manifest['timestamp_utc'],
     end_timestamp=end,end_timestamp_source='model_999.pt mtime after completed training',
     wall_time_seconds=elapsed_seconds,wall_time_source='rounded runner elapsed display; exact perf_counter duration unavailable after post-training exception',
     run_path=str(RUN.relative_to(ROOT)),selected_iteration=saved['iter'],selected_training_mean_reward=saved['infos']['mean_reward'],
     selected_checkpoint=str((BASE/'checkpoints/best_model.pt').relative_to(ROOT)),checkpoint_sha256=sha(selected),
     final_checkpoint=str((BASE/'checkpoints/final_model.pt').relative_to(ROOT)),final_checkpoint_sha256=sha(final),
     training_process_exit_code=1,post_training_metadata_recovered=True,recovery_record='training_finalization_recovery.json'))
save('training_finalization_recovery.json',dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
     failure='After training completed, relative output Path.relative_to(absolute ROOT) raised ValueError while saving checkpoint_selection.json',
     fix='Resolve output path before launcher/config processing',training_completed=True,updates_verified=1000,transitions_verified=131072000,
     evidence=['training_curve.csv','train_stdout.log','model_999.pt','best_model.pt'],
     source_used_for_training='source_snapshots/baseline_trainer_used_for_training.py',source_used_sha256=sha(BASE/'source_snapshots/baseline_trainer_used_for_training.py'),
     corrected_source_sha256=sha(ROOT/'scripts/baseline_observation_ablation_budget.py'),
     retrained=False,checkpoint_bytes_changed=False,evaluation_used_for_selection=False,raw_failure_log_preserved=True))
print('RECOVERY PASS: 1000 updates / 131072000 transitions; selected iteration',saved['iter'],'mean',saved['infos']['mean_reward'])
