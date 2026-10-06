"""Publish compact HeightScan reports and verify protected historical artifacts.

Run only after training/evaluation complete. Never writes historical directories.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/observation_ablation"
HEIGHT = EXP / "heightscan"
TRAIN = ROOT / "logs/rsl_rl/observation_ablation/ablation_heightscan_stock_s42"
RESULT = HEIGHT / "results"


def read(path):
    return json.loads(path.read_text())


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_finite(value):
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for v in value.values():
            check_finite(v)
    elif isinstance(value, list):
        for v in value:
            check_finite(v)


def main():
    import torch
    training = read(TRAIN / "training_summary.json")
    evaluation = read(RESULT / "evaluation_summary.json")
    manifest = read(TRAIN / "manifest.json")
    assert training["completed"] and training["iterations"] == 10000
    assert evaluation["first_episode_only"] and evaluation["finite"]
    assert training["checkpoint_sha256"] == read(RESULT / "manifest.json")["checkpoint_sha256"]
    assert read(HEIGHT / "smoke/smoke.json")["pass"]
    core = read(EXP / "shared/source_parity.json")
    for folder in (TRAIN, RESULT):
        actual = read(folder / "source_parity.json")
        assert actual["hashes"] == core["hashes"]
        assert actual["reward_hash"] == core["reward_hash"]
        assert actual["ppo_hash"] == core["ppo_hash"]
    training_sources = read(TRAIN / "source_mapping.json")
    evaluation_sources = read(RESULT / "source_mapping.json")
    # Evaluation-only diagnostics were corrected after raw no-hit Inf was seen.
    # The training runner, policy, environment and sensor semantics are unchanged.
    driver = str(ROOT / "scripts/observation_ablation.py")
    assert {k:v for k,v in training_sources.items() if k != driver} == {
        k:v for k,v in evaluation_sources.items() if k != driver}
    manifest["driver_source_hashes"] = {"training": training_sources[driver], "evaluation": evaluation_sources[driver]}
    manifest["driver_revision_note"] = "Post-training diagnostic fix only: accept and count native no-hit Inf, preserving canonical clipping. No training/env/reward/policy change."
    selected = Path(training["selected_checkpoint"])
    if not selected.is_absolute():
        selected = ROOT / selected
    assert selected.stat().st_size < 100 * 1024 * 1024, "Selected checkpoint needs an explicit large-file policy"
    checkpoint_dir = HEIGHT / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    portable_checkpoint = checkpoint_dir / "best_model.pt"
    shutil.copyfile(selected, portable_checkpoint)
    assert sha(portable_checkpoint) == training["checkpoint_sha256"]
    checkpoint = torch.load(portable_checkpoint, map_location="cpu", weights_only=False)
    tensor_count = 0

    def check_checkpoint(value):
        nonlocal tensor_count
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), "Nonfinite checkpoint tensor"
            tensor_count += 1
        elif isinstance(value, dict):
            for v in value.values():
                check_checkpoint(v)
        elif isinstance(value, (list, tuple)):
            for v in value:
                check_checkpoint(v)
        else:
            check_finite(value)

    check_checkpoint(checkpoint)
    training["repository_checkpoint"] = str(portable_checkpoint.relative_to(ROOT))
    training["checkpoint_bytes"] = portable_checkpoint.stat().st_size
    manifest["training"] = training
    manifest["evaluation"] = read(RESULT / "manifest.json")
    manifest["evaluation_protocol_frozen_before_training"] = True
    dump(HEIGHT / "manifest.json", manifest)
    dump(HEIGHT / "training_summary.json", training)
    dump(HEIGHT / "source_mapping.json", {"training": training_sources, "evaluation": evaluation_sources,
         "driver_revision_note": manifest["driver_revision_note"]})
    for name in ("evaluation_summary.json", "main_metrics.csv", "reward_components.csv", "episode_metrics.csv", "raycast_diagnostics.json"):
        shutil.copyfile(RESULT / name, HEIGHT / name)
    protocol = read(EXP / "protocol.json")
    protocol["team1_base_commit"] = protocol["git_commit"]
    protocol["implementation_commit"] = manifest["git_commit"]
    protocol["status"] = "HeightScan complete; Baseline stock-reward experiments pending"
    dump(EXP / "protocol.json", protocol)

    # Retain terrain geometry and assignment diagnostics from this run.
    parity = dict(frozen_config=True, reward=True, ppo=True, controlled_sources=True,
                  driver_diagnostic_fix_only=True)
    rows = list(csv.DictReader((RESULT / "episode_metrics.csv").open()))
    assert len(rows) == 100 and sorted(int(r["env_id"]) for r in rows) == list(range(100))
    config = read(RESULT / "config.json")
    terrain = config["scene"]["terrain"]["terrain_generator"]
    half_x = terrain["num_rows"] * terrain["size"][0] / 2
    x_bound = half_x + terrain["border_width"]
    initial_x_offset = config["scene"]["robot"]["init_state"]["pos"][0]
    outside = []
    for row in rows:
        origin_x = (int(row["terrain_row"]) + 0.5) * terrain["size"][0] - half_x
        terminal_x = origin_x + initial_x_offset + float(row["forward_displacement"])
        if abs(terminal_x) > x_bound:
            outside.append(int(row["env_id"]))
    rays = read(RESULT / "raycast_diagnostics.json")
    rays.update(terminal_x_outside_map_count=len(outside), terminal_x_outside_map_env_ids=outside,
                terrain_x_bounds=[-x_bound, x_bound],
                raw_missing_ray_fraction=rays["raw_missing_ray_samples"] / (sum(int(r["episode_steps"]) for r in rows) * 63),
                bounds_method="Derived from frozen native centered terrain dimensions/border and row origins; native reset has no root X randomization.")
    dump(HEIGHT / "raycast_diagnostics.json", rays)
    dump(EXP / "shared/evaluation_parity.json", parity)

    metrics = evaluation["metrics"]
    reward = evaluation["reward_components"]
    with (HEIGHT / "main_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "mean", "std", "min", "max", "count", "ratio"], lineterminator="\n")
        w.writeheader()
        w.writerows(dict(metric=k, **v) for k,v in metrics.items())
    text = (EXP / "README.md").read_text()
    text = text.replace("Pending runtime validation, training and evaluation. TBD is not a measured zero.",
                        "HeightScan training and 100-env first-episode evaluation are complete. Baseline remains TBD.")
    def fmt(v):
        return f"{v['mean']:.6f} ± {v['std']:.6f}"
    for label, key in [("Return mean ± population std", "episode_return"),
                       ("Displacement mean ± population std", "forward_displacement")]:
        text = text.replace(f"| {label} | TBD | TBD |", f"| {label} | TBD | {fmt(metrics[key])} |")
    for label, key in [("Fall", "fall"), ("Timeout", "timeout")]:
        v = metrics[key]
        text = text.replace(f"| {label} | TBD | TBD |", f"| {label} | TBD | {v['count']}/100 ({v['ratio']:.0%}) |")
    text = text.replace("episode residual tolerance=1e-3, per-step tolerance=1e-5. Component statistics are TBD.",
                        "episode residual tolerance=1e-3, per-step tolerance=1e-5.\n\n"
                        + "| Component | Baseline | HeightScan mean ± population std |\n"
                        + "|---|---:|---:|\n"
                        + "\n".join(f"| {k} | TBD | {fmt(v)} |" for k, v in reward.items())
                        + f"\n\nMax step residual={evaluation['max_step_reward_residual']:.3g}; "
                        + f"max episode residual={evaluation['max_episode_reward_residual']:.3g}.")
    if "Selected checkpoint:" not in text:
        text += ("\nSelected checkpoint: `" + training["selected_checkpoint"] + "`; saved iteration="
                 + str(training["selected_iteration"]) + "; SHA256=`" + training["checkpoint_sha256"] + "`.\n")
    if "Observed terrain-boundary caveat:" not in text:
        text += (f"\nObserved terrain-boundary caveat: {len(outside)}/100 terminal world-X positions lie outside "
                 + f"the configured terrain mesh X extent [-{x_bound:g}, {x_bound:g}] m. "
                 + f"Raw ray misses={rays['raw_missing_ray_samples']} ({rays['raw_missing_ray_fraction']:.2%} of active ray samples), "
                 + "mapped to -1 by the unchanged canonical clip. Team1 has no map-boundary or torso-height termination; "
                 + "large displacement/return therefore includes movement beyond the bounded map and must not be interpreted "
                 + "as entirely supported terrain locomotion. These dynamics/terminations were preserved for parity.\n")
    (EXP / "README.md").write_text(text)

    # Compact Git artifacts use LF; preserve the CSV field values exactly.
    for folder in (HEIGHT, RESULT):
        for path in folder.glob("*.csv"):
            path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
    for folder in (HEIGHT / "smoke", RESULT):
        path = folder / "run.log"
        if path.exists():
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")

    protected = read(EXP / "shared/protected_artifacts_before.json")
    dependencies = read(EXP / "shared/dependency_sources.json")
    for name, expected in dependencies["sources"].items():
        assert sha(Path("/home/zxro/IsaacLab_RS") / name) == expected, name
    changed = [name for name, record in protected.items()
               if not (ROOT / name).is_file() or (ROOT / name).stat().st_size != record["size"]
               or sha(ROOT / name) != record["sha256"]]
    assert not changed, changed
    pyfiles = [ROOT / name for name in ["source/ant/__init__.py", "source/ant/ablation_env_cfg.py",
               "source/ant/agents/ablation_ppo_cfg.py", "scripts/observation_ablation.py",
               "scripts/finalize_observation_ablation.py"]]
    subprocess.run(["python", "-m", "py_compile", *map(str, pyfiles)], check=True, cwd=ROOT)
    subprocess.run(["git", "diff", "--check"], check=True, cwd=ROOT)
    json_count = csv_count = 0
    for path in EXP.rglob("*.json"):
        check_finite(read(path))
        json_count += 1
    for path in EXP.rglob("*.csv"):
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                assert None not in row
                for v in row.values():
                    try:
                        numeric = float(v)
                    except (ValueError, TypeError):
                        continue
                    assert math.isfinite(numeric), (path, v)
        csv_count += 1
    report = dict(phase="complete", pass_=True, protected_files=len(protected),
                  protected_sha256_unchanged=True, existing_checkpoints_unchanged=True,
                  py_compile=True, git_diff_check=True, json_parse=True, json_files=json_count,
                  csv_parse=True, csv_files=csv_count, nan_inf_check=True, smoke_pass=True,
                  selected_checkpoint_finite=True, checkpoint_tensors=tensor_count,
                  canonical_dependency_sources_unchanged=True,
                  raw_ray_miss_handling="canonical clip to -1; processed observations/actions/rewards finite",
                  raw_missing_ray_samples=rays["raw_missing_ray_samples"],
                  terminal_x_outside_map_count=len(outside),
                  training_complete=True, evaluation_complete=True, runtime_source_parity=parity,
                  step_residual=evaluation["max_step_reward_residual"], episode_residual=evaluation["max_episode_reward_residual"])
    dump(HEIGHT / "integrity.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
