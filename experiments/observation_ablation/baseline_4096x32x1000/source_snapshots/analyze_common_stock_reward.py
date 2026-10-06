"""Validate/recompute the separate common-objective evaluation, without rollouts.

Uses population standard deviations, like canonical evaluators. Paired differences
are descriptive only; no training-seed significance test or new bootstrap method.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/observation_ablation/common_stock_reward_evaluation"
POLICIES = ("heightscan_stock", "heightscan_contact_stock", "heightscan_contact_modified_trained")
CONTINUOUS = ("episode_return", "forward_displacement", "episode_duration", "mean_forward_velocity")
TERMS = ("progress", "alive", "upright", "move_to_target", "action_l2", "energy", "joint_pos_limits")
CANONICAL = ("heightscan_4096x32x1000", "heightscan_contact_stock_4096x32x1000", "heightscan_contact_modified_4096x32x1000")
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--check-only", action="store_true", help="Validate existing results without writing outputs")
args = parser.parse_args()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def stats(values):
    return dict(mean=statistics.mean(values), std=statistics.pstdev(values))


def dump(name, value):
    if args.check_only:
        return
    path = OUTPUT / name
    assert not path.exists(), path
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


episodes, manifests, summaries, initials = {}, {}, {}, {}
for policy in POLICIES:
    folder = OUTPUT / policy
    summaries[policy] = summary = read(folder / "summary.json")
    manifests[policy] = manifest = read(folder / "manifest.json")
    initials[policy] = read(folder / "initial_conditions.json")
    with (folder / "episodes.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 100 and [int(r["env_id"]) for r in rows] == list(range(100))
    assert summary["completed_first_episodes"] == 100 and summary["finite"]
    assert summary["first_episode_only"] and summary["terminal_step_included"] and summary["post_reset_rewards_excluded"]
    assert manifest["evaluation_reward"] == "Stock" and manifest["evaluation_seed"] == 24 and manifest["terrain_seed"] == 42
    assert manifest["num_envs"] == 100 and manifest["deterministic_mean_actions"]
    assert manifest["observation_dimension"] == (122 if policy == "heightscan_stock" else 126)
    assert sha(ROOT / manifest["checkpoint"]) == manifest["checkpoint_sha256"]
    for row in rows:
        for name, value in row.items():
            if value not in ("True", "False"):
                assert math.isfinite(float(value)), (policy, name)
        assert 1 <= int(row["episode_steps"]) <= 960
        assert math.isclose(int(row["episode_steps"]) / 60, float(row["episode_duration"]), abs_tol=1e-12)
        assert row["fall"] == "True" or row["timeout"] == "True"
        assert math.isclose(sum(float(row["reward_" + name]) for name in TERMS), float(row["episode_return"]), abs_tol=1e-3)
    for metric in CONTINUOUS:
        calculated = stats([float(row[metric]) for row in rows])
        for statistic, value in calculated.items():
            assert math.isclose(value, summary["metrics"][metric][statistic], abs_tol=1e-11)
    for metric in ("fall", "timeout", "other", "out_of_terrain_x"):
        count = sum(r[metric] == "True" for r in rows)
        assert summary["metrics"][metric] == dict(count=count, ratio=count/100)
    for distance in (2, 5, 10):
        count = sum(float(r["forward_displacement"]) >= distance for r in rows)
        assert summary["metrics"][f">={distance}m"] == dict(count=count, ratio=count/100)
    episodes[policy] = rows

reference = manifests[POLICIES[0]]
for policy in POLICIES[1:]:
    for key in ("runtime_reward_terms", "active_evaluation_reward_config", "termination_config", "reset_events", "terrain_mesh_sha256"):
        assert manifests[policy][key] == reference[key], (policy, key)
    assert initials[policy] == initials[POLICIES[0]], "Initial conditions are not paired"

canonical_parity = {}
for policy, canonical in zip(POLICIES, CANONICAL):
    folder = ROOT / "experiments/observation_ablation" / canonical
    # Historical evaluation layout is intentionally not changed.
    candidates = list(folder.rglob("episode_metrics.csv"))
    assert candidates
    matches = []
    for path in candidates:
        with path.open() as stream:
            previous = list(csv.DictReader(stream))
        if len(previous) != 100:
            continue
        fields = ("initial_world_x", "terrain_row", "terrain_column", "episode_steps", "forward_displacement", "mean_forward_velocity", "fall", "timeout")
        match = all(old[key] == new[key] for old, new in zip(previous, episodes[policy]) for key in fields)
        matches.append(dict(path=str(path.relative_to(ROOT)), physical_metrics_and_initial_assignment_identical=match))
    canonical_parity[policy] = matches

pairs = {}
for left, right in zip(POLICIES, POLICIES[1:]):
    differences = {metric: stats([float(b[metric]) - float(a[metric]) for a, b in zip(episodes[left], episodes[right])]) for metric in CONTINUOUS}
    for metric in ("fall", "timeout"):
        differences[metric + "_count_change"] = summaries[right]["metrics"][metric]["count"] - summaries[left]["metrics"][metric]["count"]
    pairs[right + "_minus_" + left] = differences

before = read(OUTPUT / "integrity_before.json")
changed = [path for path, value in before.items() if not (ROOT / path).is_file() or sha(ROOT / path) != value]
documentation = "experiments/observation_ablation/README.md"
assert not [path for path in changed if path != documentation], changed
if changed:
    final_record = read(OUTPUT / "final_integrity.json")
    assert sha(ROOT / documentation) == final_record["intentional_documentation_changes"][documentation]["after_sha256"]
dump("comparison.json", dict(evaluation_reward="Stock for all policies", summaries=summaries, paired=True,
     pairing_basis="Identical full initial root state, joint position/velocity, origins, terrain rows/columns, material properties and terrain mesh hash",
     paired_descriptive_differences=pairs, canonical_behavior_parity=canonical_parity,
     single_training_seed=True, statistical_significance_claimed=False))
dump("integrity_validation.json", dict(pass_=True, protected_existing_files=len(before), protected_existing_files_unchanged=True,
     checkpoint_hashes_unchanged=True, first_episodes_per_policy=100, unique_complete_environment_ids=True,
     finite=True, summary_recomputed_from_csv=True, common_reward_verified=True, full_initial_conditions_paired=True,
     note="Before README additions; final preservation record distinguishes the intentional experiment README addition."))
if not args.check_only:
    stream = (OUTPUT / "comparison.csv").open("w", newline="")
else:
    import io
    stream = io.StringIO()
with stream:
    columns = ["training_condition", "evaluation_reward"] + [metric + "_" + stat for metric in CONTINUOUS for stat in ("mean", "std")] + ["fall_count", "timeout_count", "reach_5m_count"]
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for policy in POLICIES:
        metrics = summaries[policy]["metrics"]
        row = dict(training_condition=policy, evaluation_reward="Stock")
        row.update({metric + "_" + stat: metrics[metric][stat] for metric in CONTINUOUS for stat in ("mean", "std")})
        row.update(fall_count=metrics["fall"]["count"], timeout_count=metrics["timeout"]["count"], reach_5m_count=metrics[">=5m"]["count"])
        writer.writerow(row)
print(json.dumps({policy: summaries[policy]["metrics"] for policy in POLICIES}, indent=2))
print("Validation PASS:", len(before) - len(changed), "existing files unchanged; intentional documentation additions:", len(changed), "; 100/100 episodes each; full pairing verified")
