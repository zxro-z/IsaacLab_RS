"""Validate measured Baseline and derive a NEW four-policy common-objective record.

Original three-policy comparison/results/manifests are read only. All statistics
use the existing population-std convention; no new significance/CI procedure.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/observation_ablation/baseline_4096x32x1000"
COMMON = ROOT / "experiments/observation_ablation/common_stock_reward_evaluation"
OUTPUT = COMMON / "four_policy_comparison"
KEYS = ("baseline", "heightscan_stock", "heightscan_contact_stock", "heightscan_contact_modified_trained")
LABELS = ("Baseline + Stock", "HeightScan + Stock", "HeightScan + Contact + Stock", "HeightScan + Contact + Modified-trained")
DIMENSIONS = (59, 122, 126, 126)
CONTINUOUS = ("episode_return", "forward_displacement", "episode_duration", "mean_forward_velocity")
TERMS = ("progress", "alive", "upright", "move_to_target", "action_l2", "energy", "joint_pos_limits")
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--check-only", action="store_true")
args = parser.parse_args()


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    assert all(math.isfinite(v) for v in values)
    return dict(mean=statistics.mean(values), std=statistics.pstdev(values))


def save(name, value):
    path = OUTPUT / name
    assert not path.exists(), path
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


train = read(BASE / "training_summary.json")
selection = read(BASE / "checkpoint_selection.json")
assert train["completed"] and train["actual_completed_iterations"] == 1000 and train["actual_total_transitions"] == 131072000
assert train["training_seed"] == train["terrain_seed"] == 42 and train["resume"] is False
with (ROOT / selection["curve"]).open() as stream:
    curve = list(csv.DictReader(stream))
assert len(curve) == 1000 and [int(r["iteration"]) for r in curve] == list(range(1000))
assert all(math.isfinite(float(r["mean_reward"])) for r in curve if r["mean_reward"])
best = max((r for r in curve if r["mean_reward"]), key=lambda r: float(r["mean_reward"]))
assert int(best["iteration"]) == selection["selected_iteration"] == train["selected_iteration"]
assert float(best["mean_reward"]) == selection["selected_training_mean_reward"] == train["selected_training_mean_reward"]
assert selection["evaluation_not_used"]
finite_checkpoints = []
for path in sorted((ROOT / train["run_path"]).glob("*.pt")):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert all(torch.isfinite(t).all() for t in saved["model_state_dict"].values())
    assert saved["model_state_dict"]["actor.0.weight"].shape[1] == 59
    finite_checkpoints.append(str(path.relative_to(ROOT)))
assert sha(ROOT / train["selected_checkpoint"]) == train["checkpoint_sha256"]
assert sha(ROOT / train["final_checkpoint"]) == train["final_checkpoint_sha256"]

summaries, manifests, episodes, initial = {}, {}, {}, {}
for key, dimension in zip(KEYS, DIMENSIONS):
    folder = COMMON / key
    summaries[key] = summary = read(folder / "summary.json")
    manifests[key] = manifest = read(folder / "manifest.json")
    initial[key] = read(folder / "initial_conditions.json")
    with (folder / "episodes.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 100 and [int(r["env_id"]) for r in rows] == list(range(100))
    assert summary["completed_first_episodes"] == 100 and summary["finite"]
    assert manifest["observation_dimension"] == dimension and manifest["evaluation_reward"] == "Stock"
    assert manifest["evaluation_seed"] == 24 and manifest["terrain_seed"] == 42 and manifest["num_envs"] == 100
    assert manifest["episode_length_s"] == 16 and manifest["max_control_steps"] == 960 and manifest["control_dt"] == 1/60
    assert manifest["deterministic_mean_actions"] and summary["first_episode_only"] and summary["terminal_step_included"] and summary["post_reset_rewards_excluded"]
    assert sha(ROOT / manifest["checkpoint"]) == manifest["checkpoint_sha256"]
    for row in rows:
        assert all(math.isfinite(float(v)) for v in row.values() if v not in ("True", "False"))
        assert 1 <= int(row["episode_steps"]) <= 960
        assert math.isclose(float(row["episode_duration"]), int(row["episode_steps"])/60, abs_tol=1e-12)
        assert row["fall"] == "True" or row["timeout"] == "True"
        assert math.isclose(sum(float(row["reward_" + term]) for term in TERMS), float(row["episode_return"]), abs_tol=1e-3)
    for metric in CONTINUOUS:
        for statistic, value in stats([float(r[metric]) for r in rows]).items():
            assert math.isclose(value, summary["metrics"][metric][statistic], abs_tol=1e-11)
    for metric in ("fall", "timeout", "other", "out_of_terrain_x"):
        count = sum(r[metric] == "True" for r in rows)
        assert summary["metrics"][metric] == dict(count=count, ratio=count/100)
    for distance in (2, 5, 10):
        count = sum(float(r["forward_displacement"]) >= distance for r in rows)
        assert summary["metrics"][f">={distance}m"] == dict(count=count, ratio=count/100)
    for term in TERMS:
        for statistic, value in stats([float(r["reward_" + term]) for r in rows]).items():
            assert math.isclose(value, summary["reward_components"][term][statistic], abs_tol=1e-11)
    episodes[key] = rows

pairing_details = {}
for key in KEYS[1:]:
    for field in ("runtime_reward_terms", "active_evaluation_reward_config", "termination_config", "reset_events", "terrain_mesh_sha256"):
        assert manifests[key][field] == manifests["baseline"][field], (key, field)
    pairing_details[key] = {field: initial[key][field] == initial["baseline"][field] for field in initial["baseline"]}
    # Equal seeds/reset distributions do not guarantee identical samples when
    # the Baseline lacks the RayCaster's zero-drift RNG draws. Record this rather
    # than changing native resets, adding a sensor, or claiming full pairing.
    for field in ("root_state_w", "env_origins", "terrain_rows", "terrain_columns", "material_properties"):
        assert pairing_details[key][field], (key, field)
retained_paired = all(initial[key] == initial[KEYS[1]] for key in KEYS[2:])
assert retained_paired
assert datetime.fromisoformat(train["end_timestamp"]) < datetime.fromisoformat(manifests["baseline"]["timestamp_utc"])
deltas = {}
for left, right in zip(KEYS, KEYS[1:]):
    paired = initial[left] == initial[right]
    if paired:
        difference = {metric: stats([float(b[metric])-float(a[metric]) for a,b in zip(episodes[left],episodes[right])]) for metric in CONTINUOUS}
    else:
        difference = {metric: dict(mean=summaries[right]["metrics"][metric]["mean"]-summaries[left]["metrics"][metric]["mean"]) for metric in CONTINUOUS}
    difference["paired_conditions_verified"] = paired
    for metric in ("fall", "timeout", ">=5m"):
        difference[metric + "_count_change"] = summaries[right]["metrics"][metric]["count"] - summaries[left]["metrics"][metric]["count"]
    deltas[right + "_minus_" + left] = difference
comparison = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), evaluation_reward="Stock for all four policies", summaries=summaries,
                  observation_dimensions=dict(zip(KEYS,DIMENSIONS)), paired=all(all(v.values()) for v in pairing_details.values()),
                  pairing_details_vs_baseline=pairing_details, retained_three_policies_paired=retained_paired,
                  pairing_basis="Fieldwise initial-state equality; same terrain/friction/root but Baseline joint samples may differ; no sensor or reset normalization added",
                  stage_deltas=deltas, single_training_seed=True, statistical_significance_claimed=False, prior_three_policy_results_rerun=False)
if not args.check_only:
    assert not OUTPUT.exists(), OUTPUT
    OUTPUT.mkdir()
    save("comparison.json", comparison)
    save("validation.json", dict(pass_=True, first_episodes_per_policy=100, unique_environment_ids=True, finite=True, summary_and_components_recomputed=True,
         observation_dimensions=list(DIMENSIONS), exact_initial_conditions_pairing=comparison["paired"], pairing_details_vs_baseline=pairing_details,
         retained_three_policies_paired=retained_paired, reward_termination_reset_parity=True,
         training_budget_verified=True, training_selection_verified=True, finite_training_checkpoints=finite_checkpoints, checkpoint_hashes_unchanged=True))
    with (OUTPUT / "comparison.csv").open("w",newline="") as stream:
        columns=["training_condition","input_dimension","evaluation_reward"]+[metric+"_"+s for metric in CONTINUOUS for s in ("mean","std")]+["fall_count","timeout_count","reach_5m_count"]
        writer=csv.DictWriter(stream,fieldnames=columns,lineterminator="\n");writer.writeheader()
        for key,label,dimension in zip(KEYS,LABELS,DIMENSIONS):
            metrics=summaries[key]["metrics"]
            row=dict(training_condition=label,input_dimension=dimension,evaluation_reward="Stock")
            row.update({metric+"_"+s:metrics[metric][s] for metric in CONTINUOUS for s in ("mean","std")})
            row.update(fall_count=metrics["fall"]["count"],timeout_count=metrics["timeout"]["count"],reach_5m_count=metrics[">=5m"]["count"])
            writer.writerow(row)
else:
    existing=read(OUTPUT/"comparison.json")
    for field in ("summaries","observation_dimensions","stage_deltas","paired"):
        assert existing[field]==comparison[field]
print(json.dumps(dict(baseline=summaries['baseline']['metrics'],stage_deltas=deltas),indent=2))
print("PASS: four policies; 100 first episodes each; 59/122/126/126; full pairing:",comparison["paired"],"; checkpoint selection/budget verified")
