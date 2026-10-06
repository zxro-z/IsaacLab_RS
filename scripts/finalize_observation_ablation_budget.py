"""Finalize the new budget run without writing any historical result directory."""
import csv
import hashlib
import json
import math
import re
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'experiments/observation_ablation'
OUT = EXP / 'heightscan_4096x32x1000'
TRAIN = ROOT / 'logs/rsl_rl/observation_ablation/ablation_heightscan_stock_s42_e4096_n32_i1000'
RESULT = OUT / 'results'


def read(p):
    return json.loads(p.read_text())


def dump(p, value):
    p.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8388608), b''):
            h.update(b)
    return h.hexdigest()


def finite(v):
    if isinstance(v, float):
        assert math.isfinite(v)
    elif isinstance(v, dict):
        for x in v.values(): finite(x)
    elif isinstance(v, (list, tuple)):
        for x in v: finite(x)


def main():
    import torch
    t = read(TRAIN / 'training_summary.json')
    e = read(RESULT / 'evaluation_summary.json')
    m = read(TRAIN / 'manifest.json')
    assert t['completed'] and t['iterations'] == 1000 and t['total_transitions'] == 131072000
    log = (TRAIN / 'run.log').read_text()
    totals = re.findall(r'Total timesteps:\s+(\d+)', log)
    iterations = re.findall(r'Learning iteration (\d+)/(\d+)', log)
    assert int(totals[-1]) == 131072000 and len(iterations) == 1000 and iterations[-1] == ('999','1000')
    t['actual_total_transitions'] = int(totals[-1])
    t['actual_completed_iterations'] = len(iterations)
    assert not t['resume'] and t['load_run'] is None and t['load_checkpoint'] is None
    assert e['finite'] and e['first_episode_only'] and read(OUT / 'smoke_verified/smoke.json')['pass']
    assert t['checkpoint_sha256'] == read(RESULT / 'manifest.json')['checkpoint_sha256']
    core = read(EXP / 'shared/source_parity.json')
    for folder in [OUT / 'smoke_verified', TRAIN, RESULT]:
        p = read(folder / 'source_parity.json')
        assert p['hashes'] == core['hashes'] and p['reward_hash'] == core['reward_hash'] and p['ppo_hash'] == core['ppo_hash']
    dependencies = read(EXP / 'shared/dependency_sources.json')
    for name, expected in dependencies['sources'].items():
        assert sha(Path('/home/zxro/IsaacLab_RS') / name) == expected, name
    sources = read(TRAIN / 'source_mapping.json')
    assert sources == read(RESULT / 'source_mapping.json') == read(OUT / 'smoke_verified/source_mapping.json')
    rows = list(csv.DictReader((RESULT / 'episode_metrics.csv').open()))
    oldrows = list(csv.DictReader((EXP / 'heightscan/episode_metrics.csv').open()))
    assert len(rows) == 100
    assert [(r['terrain_row'], r['terrain_column']) for r in rows] == [(r['terrain_row'], r['terrain_column']) for r in oldrows]
    oldeval = read(EXP / 'heightscan/evaluation_summary.json')
    assert oldeval['terrain_mesh_sha256'] == e['terrain_mesh_sha256']
    assert e['terrain_x_bounds'] == [-102.0, 102.0]
    tensor_count = 0
    def check(v):
        nonlocal tensor_count
        if isinstance(v, torch.Tensor):
            assert torch.isfinite(v).all()
            tensor_count += 1
        elif isinstance(v, dict):
            for x in v.values(): check(x)
        elif isinstance(v, (list, tuple)):
            for x in v: check(x)
        else: finite(v)
    (OUT / 'checkpoints').mkdir(exist_ok=True)
    records = {}
    for name, source in [('best_model.pt', TRAIN / 'best_model.pt'), ('final_model.pt', TRAIN / 'model_999.pt')]:
        assert source.stat().st_size < 100 * 1024 * 1024
        target = OUT / 'checkpoints' / name
        shutil.copyfile(source, target)
        assert sha(source) == sha(target)
        data = torch.load(target, map_location='cpu', weights_only=False)
        check(data)
        records[name] = dict(path=str(target.relative_to(ROOT)), sha256=sha(target), bytes=target.stat().st_size, saved_iteration=data['iter'])
    assert records['best_model.pt']['sha256'] == t['checkpoint_sha256']
    t['repository_checkpoint'] = records['best_model.pt']['path']
    t['repository_final_checkpoint'] = records['final_model.pt']['path']
    m.update(training=t, evaluation=read(RESULT / 'manifest.json'), canonical=True, budget_only_change=True,
             environment_reward_ppo_observation_parity=True, evaluation_terrain_assignment_parity=True)
    dump(OUT / 'manifest.json', m)
    dump(OUT / 'training_summary.json', t)
    dump(OUT / 'checkpoint_selection.json', dict(rule=t['selection'], fixed_before_training=True,
         selected_training_mean_return=t['selected_training_mean_reward'], checkpoints=records))
    dump(OUT / 'source_mapping.json', dict(training=sources, evaluation=sources, canonical_dependencies=dependencies, finalizer_sha256=sha(Path(__file__))))
    for name in ['evaluation_summary.json','reward_components.csv','episode_metrics.csv','raycast_diagnostics.json']:
        shutil.copyfile(RESULT / name, OUT / name)
    with (OUT / 'main_metrics.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['metric','mean','std','min','max','count','ratio'], lineterminator='\n')
        w.writeheader(); w.writerows(dict(metric=k, **v) for k,v in e['metrics'].items())
    dump(OUT / 'parity.json', dict(environment=True, reward=True, ppo=True, architecture=True,
         observation=True, terrain_mesh=True, evaluation_terrain_assignment=True, budget_only_change=True))
    p = read(EXP / 'protocol.json')
    p.update(status='Canonical HeightScan complete; matching-budget baseline result unavailable',
             git_commit=m['git_commit'], implementation_commit=m['git_commit'], canonical_heightscan='heightscan_4096x32x1000',
             canonical_checkpoint=records['best_model.pt']['path'])
    dump(EXP / 'protocol.json', p)
    metrics = e['metrics']; rewards = e['reward_components']
    def fmt(v): return f"{v['mean']:.4f} ± {v['std']:.4f}"
    readme = documentation()
    readme += '\n## Results\n\n| Metric | Baseline | HeightScan |\n|---|---:|---:|\n'
    for label, key in [('Return','episode_return'),('Displacement (m)','forward_displacement'),('Duration (s)','episode_duration'),('Mean vx (m/s)','mean_forward_velocity')]:
        readme += f'| {label} | TBD | {fmt(metrics[key])} |\n'
    for label, key in [('Fall','fall'),('Timeout','timeout'),('Other','other'),('>=5m','>=5m'),('Out-of-terrain-X','out_of_terrain_x')]:
        readme += f"| {label} | TBD | {metrics[key]['count']}/100 |\n"
    readme += '\n## Stock reward decomposition\n\n| Component | HeightScan mean ± population std |\n|---|---:|\n'
    for k,v in rewards.items(): readme += f'| {k} | {fmt(v)} |\n'
    readme += '\n## Training-budget sensitivity (diagnostic only)\n\n| Metric | Previous: 2048×32×10000 | Canonical: 4096×32×1000 |\n|---|---:|---:|\n'
    readme += '| Transitions | 655,360,000 | 131,072,000 |\n'
    readme += f"| Selected saved iteration | 4911 | {t['selected_iteration']} |\n"
    for label,key in [('Return','episode_return'),('Displacement','forward_displacement')]:
        readme += f'| {label} | {fmt(oldeval["metrics"][key])} | {fmt(metrics[key])} |\n'
    for label,key,old in [('Fall','fall','35%'),('Timeout','timeout','65%'),('>=5m','>=5m','86%'),('Out-of-terrain-X','out_of_terrain_x','27%')]:
        readme += f"| {label} | {old} | {metrics[key]['count']}% |\n"
    readme += '\nThe new run has five times fewer transitions. This is training-budget sensitivity, not an isolated environment-count effect or the observation-ablation result.\n'
    readme += f"\nTerrain boundary diagnostic: {metrics['out_of_terrain_x']['count']}/100 terminal world-X positions outside [-102, 102] m. No boundary or torso-height termination exists; displacement/return can include unsupported movement beyond the map. Raw ray misses retain the canonical clipping to -1.\n"
    (EXP / 'README.md').write_text(readme)
    for folder in [OUT / 'smoke_verified', RESULT]:
        log = folder / 'run.log'
        if log.exists(): log.write_text('\n'.join(x.rstrip() for x in log.read_text().splitlines())+'\n')
    protected = read(OUT / 'protected_artifacts_before.json')
    changed = [name for name,r in protected.items() if not (ROOT/name).is_file() or (ROOT/name).stat().st_size != r['bytes'] or sha(ROOT/name) != r['sha256']]
    assert not changed, changed
    pyfiles = ['scripts/observation_ablation_budget.py','scripts/finalize_observation_ablation_budget.py','source/ant/agents/ablation_budget_ppo_cfg.py']
    subprocess.run(['python','-m','py_compile',*pyfiles],cwd=ROOT,check=True)
    subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
    nj = nc = 0
    for path in OUT.rglob('*.json'): finite(read(path)); nj += 1
    for path in OUT.rglob('*.csv'):
        with path.open(newline='') as f:
            for row in csv.DictReader(f):
                assert None not in row
                for v in row.values():
                    try: x=float(v)
                    except (ValueError,TypeError): continue
                    assert math.isfinite(x)
        nc += 1
    dump(OUT / 'integrity.json', dict(pass_=True, py_compile=True, git_diff_check=True, json_parse=True, csv_parse=True,
         nan_inf_check=True, json_files=nj,csv_files=nc, checkpoint_tensors_finite=True, checkpoint_tensor_count=tensor_count,
         protected_files=len(protected),protected_sha256_unchanged=True, existing_checkpoints_unchanged=True,
         source_parity=True,terrain_mesh_assignment_parity=True, max_step_residual=e['max_step_reward_residual'],
         max_episode_residual=e['max_episode_reward_residual'],mean_episode_residual=e['mean_episode_reward_residual']))
    (OUT / 'files_added.txt').write_text('\n'.join(sorted({str(p.relative_to(ROOT)) for p in OUT.rglob('*') if p.is_file() and 'smoke' not in p.relative_to(OUT).parts} | {str((OUT / 'files_added.txt').relative_to(ROOT))}))+'\n')
    print(json.dumps(t,indent=2)); print(json.dumps(e,indent=2))


def documentation():
    return '''# Observation Ablation

## Research question

Does explicit local terrain-height information improve locomotion over uneven terrain? Compare the stock-observation baseline with HeightScan while keeping stock rewards and training conditions fixed. Matching-budget baseline results are unavailable; the HeightScan measurements alone do not establish improvement.

## Canonical protocol

HeightScan uses 4096 environments, 32 steps per rollout, 1000 updates (131,072,000 transitions), training/terrain seed42, [400,200,100] ELU actor/critic, stock rewards and seed24/100-env deterministic evaluation. Baseline must match these settings before a controlled Stage 1 comparison is claimed.

## Stock reward

| Term | Weight |
|---|---:|
| progress | 1.0 |
| alive | 0.5 |
| upright | 0.1 |
| move_to_target | 0.5 |
| action_l2 | -0.005 |
| energy | -0.05 |
| joint_pos_limits | -0.1 |

Canonical IsaacLab_RS Ant reward functions are reused directly. Team1 TotalReward and foot_contact / joint_velocity / foot_slip rewards are inactive.

## Environment and observation

Team1 five equal-proportion rough terrains, 20×10 patches of 10×10 m, terrain seed42. Physics dt=1/120 s, decimation2, control dt=1/60 s. Eight joint-effort actions, scale7.5. Native root reset, joint position ±0.2 / velocity ±0.1. Timeout16 s (960 steps) or body_z_down(pi/2); no boundary/torso-height termination.

HeightScan integration is adapted from the IsaacLab_RS Assignment 1 HeightScan implementation. Native proprio59 + scan63 = actor/critic122. Torso attached, yaw aligned, offset(0.8,0,20), 9×7 downward rays spaced0.2 m; sensor-Z − hit-Z −0.5, scale1, clip[-1,1]. No explicit Contact feature, CNN, dummy feature or empirical observation normalization. Existing incoming foot wrench is retained.

## Training protocol

Fresh seed42 initialization; resume=false, load_run=null, load_checkpoint=null. New driver: `scripts/observation_ablation_budget.py`; derived runner: `source/ant/agents/ablation_budget_ppo_cfg.py`. Historical task/config and driver remain intact. Only num_envs/max_iterations budget changes; rollout32 remains unchanged. The canonical protocol is [protocol.json](protocol.json).

PPO unchanged: lr0.0005 adaptive, gamma0.99, lambda0.95, clip0.2, entropy0, value loss1, clipped value loss, 5 epochs / 4 minibatches, desired KL0.01, grad norm1. Actor/critic [400,200,100] ELU; init Gaussian noise std1; actor/critic observation normalization false. Save interval50. Checkpoint selection, fixed before training: highest logged completed-episode training mean return. Final checkpoint is also retained.

## Evaluation protocol

Team1 environment, seed24, 100 envs, deterministic mean action, first episode only, stock7 terms; include terminal rewards and exclude subsequent reset episodes. Population standard deviation. Forward displacement is terminal minus initial world-X. Reward components accumulate actual RewardManager contributions (raw×weight×control_dt); residuals checked against official return. Terrain mesh and assignment must match the previous run.

## Artifact policy and caveats

Canonical HeightScan artifacts: [heightscan_4096x32x1000/manifest.json](heightscan_4096x32x1000/manifest.json). Previous exploratory run: [heightscan/manifest.json](heightscan/manifest.json), 2048×32×10000; preserved byte-for-byte. Large raw logs/intermediate checkpoints remain in ignored logs; compact JSON/CSV and selected/final checkpoints (<100 MiB each) are committed.

Single-seed results do not establish general statistical significance; PPO remains stochastic. All shared settings must match for direct comparison.
'''


if __name__ == '__main__':
    main()
