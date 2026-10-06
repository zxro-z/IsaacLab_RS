"""Validate and package Contact+Stock without modifying historical artifacts."""
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'experiments/observation_ablation'
OUT = EXP / 'heightscan_contact_stock_4096x32x1000'
TRAIN = ROOT / 'logs/rsl_rl/observation_ablation/ablation_heightscan_contact_stock_s42_e4096_n32_i1000'
RESULT = OUT / 'results'
BASE = EXP / 'heightscan_4096x32x1000'
PYFILES = ['scripts/contact_stock_observation_ablation.py', 'scripts/finalize_contact_stock_observation_ablation.py',
           'source/ant/contact_stock_env_cfg.py', 'source/ant/agents/contact_stock_ppo_cfg.py', 'source/ant/__init__.py']


def read(path):
    return json.loads(path.read_text())


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8388608), b''):
            h.update(chunk)
    return h.hexdigest()


def finite(value):
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for x in value.values(): finite(x)
    elif isinstance(value, (list, tuple)):
        for x in value: finite(x)


def main():
    import torch
    t = read(TRAIN / 'training_summary.json')
    e = read(RESULT / 'evaluation_summary.json')
    m = read(TRAIN / 'manifest.json')
    smoke = read(OUT / 'smoke/smoke.json')
    assert smoke['pass'] and smoke['stock_reward_parity'] and not smoke['custom_reward_active']
    assert t['completed'] and t['iterations'] == 1000 and t['total_transitions'] == 131072000
    assert not t['resume'] and t['load_run'] is None and t['load_checkpoint'] is None
    log = (OUT / 'training_run.log').read_text()
    totals = re.findall(r'Total timesteps:\s+(\d+)', log)
    iterations = re.findall(r'Learning iteration (\d+)/(\d+)', log)
    assert int(totals[-1]) == 131072000 and len(iterations) == 1000 and iterations[-1] == ('999','1000')
    t.update(actual_total_transitions=int(totals[-1]), actual_completed_iterations=len(iterations))
    assert e['finite'] and e['first_episode_only'] and e['terminal_step_included'] and e['post_reset_rewards_excluded']
    assert t['checkpoint_sha256'] == read(RESULT / 'manifest.json')['checkpoint_sha256']
    core = read(BASE / 'results/source_parity.json')
    sources = read(TRAIN / 'source_mapping.json')
    for folder in [OUT / 'smoke', TRAIN, RESULT]:
        p = read(folder / 'source_parity.json')
        assert p['hashes'] == core['hashes'] and p['reward_hash'] == core['reward_hash'] and p['ppo_hash'] == core['ppo_hash']
        assert read(folder / 'source_mapping.json') == sources
        cfg = read(folder / 'config.json')
        baseline = read(BASE / 'results/config.json')
        for key in ['sim','actions','events','terminations','episode_length_s','decimation','rewards']:
            assert cfg[key] == baseline[key], key
        for key in ['terrain','robot','height_scanner','contact_forces','clone_in_fabric']:
            assert cfg['scene'][key] == baseline['scene'][key], key
        obs = cfg['observations']['policy'].copy()
        obs.pop('foot_contacts')
        assert obs == baseline['observations']['policy']
        agent = read(folder / 'agent.json'); oldagent = read(BASE / 'results/agent.json')
        for key in ['policy','algorithm','num_steps_per_env','max_iterations','seed','obs_groups']:
            assert agent[key] == oldagent[key], key
    for name, expected in sources.items():
        assert sha(Path(name)) == expected, name
    assert read(TRAIN / 'config.json')['scene']['num_envs'] == 4096
    rows = list(csv.DictReader((RESULT / 'episode_metrics.csv').open()))
    oldrows = list(csv.DictReader((BASE / 'episode_metrics.csv').open()))
    assert len(rows) == 100
    for keys in [('terrain_row','terrain_column'),('initial_world_x',)]:
        assert [tuple(r[k] for k in keys) for r in rows] == [tuple(r[k] for k in keys) for r in oldrows], keys
    assert read(BASE / 'evaluation_summary.json')['terrain_mesh_sha256'] == e['terrain_mesh_sha256']
    assert e['max_step_reward_residual'] < 1e-5 and e['max_episode_reward_residual'] < 1e-3
    tensor_count = 0
    def check(value):
        nonlocal tensor_count
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all()
            tensor_count += 1
        elif isinstance(value, dict):
            for x in value.values(): check(x)
        elif isinstance(value, (list, tuple)):
            for x in value: check(x)
        else: finite(value)
    (OUT / 'checkpoints').mkdir(exist_ok=True)
    records = {}
    for name, source in [('best_model.pt', TRAIN / 'best_model.pt'), ('final_model.pt', TRAIN / 'model_999.pt')]:
        assert source.stat().st_size < 100 * 1024 * 1024
        target = OUT / 'checkpoints' / name
        assert not target.exists()
        shutil.copyfile(source, target)
        assert sha(source) == sha(target)
        data = torch.load(target, map_location='cpu', weights_only=False)
        check(data)
        records[name] = dict(path=str(target.relative_to(ROOT)), sha256=sha(target), bytes=target.stat().st_size, saved_iteration=data['iter'])
    t.update(repository_checkpoint=records['best_model.pt']['path'], repository_final_checkpoint=records['final_model.pt']['path'])
    m.update(training=t, evaluation=read(RESULT / 'manifest.json'), contact_feedback_comparison=True,
             baseline='heightscan_4096x32x1000', changed_variable='appended explicit 4-D foot-contact observation',
             environment_reward_ppo_heightscan_parity=True, evaluation_terrain_assignment_parity=True)
    dump(OUT / 'manifest.json', m)
    dump(OUT / 'training_summary.json', t)
    dump(OUT / 'checkpoint_selection.json', dict(rule=t['selection'], fixed_before_training=True,
         selected_training_mean_return=t['selected_training_mean_reward'], checkpoints=records))
    dump(OUT / 'source_mapping.json', dict(training=sources, evaluation=sources, finalizer_sha256=sha(Path(__file__))))
    for name in ['evaluation_summary.json','reward_components.csv','episode_metrics.csv','raycast_diagnostics.json']:
        shutil.copyfile(RESULT / name, OUT / name)
    with (OUT / 'main_metrics.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['metric','mean','std','min','max','count','ratio'], lineterminator='\n')
        w.writeheader()
        w.writerows(dict(metric=k, **v) for k,v in e['metrics'].items())
    shutil.copyfile(OUT / 'smoke/contact_observation_validation.json', OUT / 'contact_observation_validation.json')
    shutil.copyfile(TRAIN / 'training_curve.csv', OUT / 'training_curve.csv')
    curve = list(csv.DictReader((OUT / 'training_curve.csv').open()))
    assert len(curve) == 1000 and [int(r['iteration']) for r in curve] == list(range(1000))
    best = max((r for r in curve if r['mean_reward']), key=lambda r:float(r['mean_reward']))
    assert int(best['iteration']) == t['selected_iteration'] and float(best['mean_reward']) == t['selected_training_mean_reward']
    old = read(BASE / 'evaluation_summary.json')
    def fmt(v): return f"{v['mean']:.4f} ± {v['std']:.4f}"
    comparison = '# Contact feedback comparison\n\nHeightScan과 Stock Reward를 고정하고 explicit 4-D Contact observation만 추가한 비교다. 동일 seed의 단일 실험이므로 일반적인 통계적 유의성을 주장하지 않는다.\n\n'
    comparison += '| Metric | HeightScan + Stock | HeightScan + Contact + Stock |\n|---|---:|---:|\n'
    for label,key in [('Return','episode_return'),('Displacement (m)','forward_displacement'),('Duration (s)','episode_duration'),('Mean vx (m/s)','mean_forward_velocity')]:
        comparison += f"| {label} | {fmt(old['metrics'][key])} | {fmt(e['metrics'][key])} |\n"
    for label,key in [('Fall','fall'),('Timeout','timeout'),('>=5m','>=5m'),('Out-of-terrain-X','out_of_terrain_x')]:
        comparison += f"| {label} | {old['metrics'][key]['count']}/100 | {e['metrics'][key]['count']}/100 |\n"
    comparison += '\n| Component | HeightScan + Stock | HeightScan + Contact + Stock |\n|---|---:|---:|\n'
    for key,value in e['reward_components'].items():
        comparison += f"| {key} | {fmt(old['reward_components'][key])} | {fmt(value)} |\n"
    comparison += '\n표준편차는 population std이다. 두 조건은 동일한 Stock Reward를 사용한다. Map-boundary termination이 없으므로 displacement와 progress에는 terrain X bounds 밖의 이동이 포함될 수 있다. 공통 README의 Stage 구조는 변경하지 않았다.\n'
    (OUT / 'comparison.md').write_text(comparison)
    # Logs are local runtime artifacts; preserve them without staging large/noisy output.
    protected = read(OUT / 'protected_artifacts_before.json')
    changed = [name for name,r in protected.items() if not (ROOT/name).is_file() or (ROOT/name).stat().st_size != r['bytes'] or sha(ROOT/name) != r['sha256']]
    assert not changed, changed
    subprocess.run(['python','-m','py_compile',*PYFILES],cwd=ROOT,check=True)
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
    files = [str(p.relative_to(ROOT)) for p in OUT.rglob('*') if p.is_file() and (p.name == 'commands.log' or not p.name.endswith('.log')) and not any(part.endswith('_attempt') for part in p.relative_to(OUT).parts)]
    files += PYFILES + [str((OUT / 'files_added.txt').relative_to(ROOT)), str((OUT / '.gitignore').relative_to(ROOT))]
    (OUT / 'files_added.txt').write_text('\n'.join(sorted(set(files)))+'\n')
    print(json.dumps(t,indent=2)); print(json.dumps(e,indent=2))


if __name__ == '__main__':
    main()
