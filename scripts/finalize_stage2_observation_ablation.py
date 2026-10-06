"""Finalize the new budget run without writing any historical result directory."""
import csv
import hashlib
import json
import math
import re
import yaml
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'experiments/observation_ablation'
OUT = EXP / 'heightscan_contact_modified_4096x32x1000'
TRAIN = ROOT / 'logs/rsl_rl/observation_ablation/ablation_heightscan_contact_modified_s42_e4096_n32_i1000'
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
    assert e['finite'] and e['first_episode_only'] and read(OUT / 'smoke/smoke.json')['pass']
    assert t['checkpoint_sha256'] == read(RESULT / 'manifest.json')['checkpoint_sha256']
    # Inspect only saved scalar config nodes: no Python-tag execution.
    saved = yaml.compose((ROOT / 'logs/rsl_rl/ant/v3_depth/params/env.yaml').read_text())
    def node_field(node, key):
        return next(value for name,value in node.value if name.value == key)
    saved_sensor = node_field(node_field(saved, 'scene'), 'contact_forces')
    actual_sensor = read(TRAIN / 'config.json')['scene']['contact_forces']
    for key in ['prim_path','history_length','force_threshold','track_air_time','update_period']:
        actual_value = actual_sensor[key]
        if key == 'prim_path':
            actual_value = actual_value.replace('{ENV_REGEX_NS}', '/World/envs/env_.*')
        assert yaml.safe_load(yaml.serialize(node_field(saved_sensor,key))) == actual_value, key
    core = read(EXP / 'shared/source_parity.json')
    for folder in [OUT / 'smoke', TRAIN, RESULT]:
        p = read(folder / 'source_parity.json')
        assert p['hashes'] == core['hashes'] and p['ppo_hash'] == core['ppo_hash']
        assert p['reward_hash'] == read(TRAIN / 'source_parity.json')['reward_hash']
    dependencies = read(EXP / 'shared/dependency_sources.json')
    for name, expected in dependencies['sources'].items():
        assert sha(Path('/home/zxro/IsaacLab_RS') / name) == expected, name
    sources = read(TRAIN / 'source_mapping.json')
    assert sources == read(RESULT / 'source_mapping.json') == read(OUT / 'smoke/source_mapping.json')
    rows = list(csv.DictReader((RESULT / 'episode_metrics.csv').open()))
    oldrows = list(csv.DictReader((EXP / 'heightscan_4096x32x1000/episode_metrics.csv').open()))
    assert len(rows) == 100
    assert [(r['terrain_row'], r['terrain_column']) for r in rows] == [(r['terrain_row'], r['terrain_column']) for r in oldrows]
    oldeval = read(EXP / 'heightscan_4096x32x1000/evaluation_summary.json')
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
    m.update(training=t, evaluation=read(RESULT / 'manifest.json'), comparison_arm='HeightScan', stage2=True,
             environment_ppo_heightscan_proprio_parity_with_stage1=True, modified_reward_parity_with_saved_v3_depth=True, reward_sensor_parity_with_saved_v3_depth=True, observation_change="appended canonical explicit 4-D contact", reward_change="Team1 v3_depth TotalReward in training and evaluation", evaluation_terrain_assignment_parity=True)
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
    dump(OUT / 'parity.json', dict(environment_with_stage1=True, modified_reward_with_saved_v3_depth=True, ppo_with_stage1=True, architecture_with_stage1=True,
         proprio_heightscan_with_stage1=True, explicit_contact_canonical=True, terrain_mesh=True, evaluation_terrain_assignment=True, stage2=True))
    for name in ['contact_observation_validation.json', 'heightscan_validation.json']:
        shutil.copyfile(OUT / 'smoke' / name, OUT / name)
    shutil.copyfile(TRAIN / 'training_curve.csv', OUT / 'training_curve.csv')
    curve = list(csv.DictReader((OUT / 'training_curve.csv').open()))
    assert len(curve) == 1000 and [int(r['iteration']) for r in curve] == list(range(1000))
    available = [r for r in curve if r['mean_reward']]
    best = max(available, key=lambda r:float(r['mean_reward']))
    assert int(best['iteration']) == t['selected_iteration'] and float(best['mean_reward']) == t['selected_training_mean_reward']
    proto_path = EXP / 'shared/stage2_contact_modified_protocol.json'
    proto = read(proto_path)
    proto.update(status='HeightScan + Contact + Modified Reward complete', implementation_commit=m['git_commit'],
                 heightscan_checkpoint=records['best_model.pt']['path'], runtime_frozen_config='heightscan_contact_modified_4096x32x1000/results/source_parity.json')
    dump(proto_path, proto)
    prefix = (EXP / 'README.md').read_text().split('## Stage 3 — Reward design')[0].rstrip()
    metrics = e['metrics']; rewards = e['reward_components']
    def fmt(v): return f"{v['mean']:.4f} ± {v['std']:.4f}"
    text = prefix + '\n\n' + documentation()
    text += '\n| Metric | HeightScan + Contact + Modified |\n|---|---:|\n'
    for label,key in [('Return','episode_return'),('Displacement (m)','forward_displacement'),('Duration (s)','episode_duration'),('Mean vx (m/s)','mean_forward_velocity')]:
        text += f'| {label} | {fmt(metrics[key])} |\n'
    for label,key in [('Fall','fall'),('Timeout','timeout'),('Other','other'),('>=5m','>=5m'),('Out-of-terrain-X','out_of_terrain_x')]:
        text += f"| {label} | {metrics[key]['count']}/100 |\n"
    text += '\n### Stage 3 reward decomposition\n\n| Component | HeightScan+Contact mean ± population std |\n|---|---:|\n'
    for key,value in rewards.items(): text += f'| {key} | {fmt(value)} |\n'
    text += f"\nResiduals: step max={e['max_step_reward_residual']:.3g}, episode max={e['max_episode_reward_residual']:.3g}, episode mean={e['mean_episode_reward_residual']:.3g}.\n"
    text += f"\nBoundary diagnostic: {metrics['out_of_terrain_x']['count']}/100 terminal world-X positions outside [-102,102] m. Native Team1 has no boundary or torso-height termination, so displacement/return can include unsupported movement beyond the terrain mesh.\n"
    text += '\nContact + Stock and Contact + Modified have different reward definitions. Do not subtract total returns to claim improvement. Physical metrics (displacement, duration, fall, timeout and reach ratios) may be inspected as descriptive diagnostics.\n'
    (EXP / 'README.md').write_text(text)
    for folder in [OUT / 'smoke', RESULT]:
        log = folder / 'run.log'
        if log.exists(): log.write_text('\n'.join(x.rstrip() for x in log.read_text().splitlines())+'\n')
    protected = read(OUT / 'protected_artifacts_before.json')
    changed = [name for name,r in protected.items() if not (ROOT/name).is_file() or (ROOT/name).stat().st_size != r['bytes'] or sha(ROOT/name) != r['sha256']]
    assert not changed, changed
    pyfiles = ['scripts/stage2_observation_ablation.py','scripts/finalize_stage2_observation_ablation.py','source/ant/stage2_env_cfg.py','source/ant/agents/stage2_ppo_cfg.py','source/ant/__init__.py']
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
    (OUT / 'files_added.txt').write_text('\n'.join(sorted({str(p.relative_to(ROOT)) for p in OUT.rglob('*') if p.is_file() and not any(part.startswith('smoke_') and part.endswith('_attempt') for part in p.relative_to(OUT).parts)} | {str((OUT / 'files_added.txt').relative_to(ROOT))}))+'\n')
    print(json.dumps(t,indent=2)); print(json.dumps(e,indent=2))


def documentation():
    return """## Stage 3 — Reward design

This is the final HeightScan + Contact configuration with modified reward. Compare it against HeightScan + Contact + Stock to study reward design.

Keep the Team1 environment, explicit 4-D contact definition, HeightScan, PPO, [400,200,100] ELU actor/critic, 4096×32×1000 budget (131,072,000 transitions), training/terrain seed42, checkpoint-selection rule, and seed24/100-env deterministic first-episode evaluation fixed. Change the reward definition. Stock and Modified returns have different scales and cannot quantify a performance improvement by subtraction.

Shared machine-readable protocol: [shared/stage2_contact_modified_protocol.json](shared/stage2_contact_modified_protocol.json). Contact spec: [shared/contact_observation.json](shared/contact_observation.json). Result manifest: [heightscan_contact_modified_4096x32x1000/manifest.json](heightscan_contact_modified_4096x32x1000/manifest.json).

HeightScan and contact observation are adapted from the existing IsaacLab_RS Assignment 1 HeightScan+Contact implementation. Observation order is Team1 proprio59, unchanged scan63, then contact4 (126-D). No CNN, dummy feature or empirical normalization. Observation contact uses current net-force norm >1N, ordered front_left_foot/front_right_foot/left_back_foot/right_back_foot, float32 binary states, history0, no clipping or scaling change.

Modified reward directly reuses Team1 v3_depth `ant.rewards.TotalReward`, manager term `total_reward` weight1. Internal weights: progress2.5, alive0.5, upright0.05, move_to_target1.5, foot_contact1, action_l2−0.005, energy−0.15, joint_velocity−0.001, joint_pos_limits−0.5, foot_slip−0.07. Reward-side contact separately uses the 3-frame maximum vertical force >5N; contact bonus requires at least two feet; slip penalizes contacted-feet XY speed. These semantics differ from observation-side contact.

Fresh initialization: resume=false, load_run/load_checkpoint=null. PPO and action/reset/termination/terrain are unchanged from Stage 1. Selection rule fixed before training: highest logged completed-episode mean training return under the modified reward. The best checkpoint and last-iteration checkpoint are retained. Single-seed results do not establish general statistical significance.

Decomposition observes the actual TotalReward function's weighted components without copying its calculations or changing its implementation. Each contribution is weighted_component × manager_weight(1) × control_dt. TotalReward episode_sums already apply dt once; dt is not applied twice. Terminal contributions are included, reset episodes excluded. Compare displacement, duration and fall/timeout under the two reward definitions; do not subtract their total returns.
"""


if __name__ == '__main__':
    main()
