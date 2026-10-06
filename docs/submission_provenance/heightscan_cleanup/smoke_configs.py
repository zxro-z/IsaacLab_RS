"""Import/config/checkpoint check only: no environments, rollout or training."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--output-dir", type=Path, default=OUT,
                    help="Write new smoke evidence here without replacing earlier reports.")
args = parser.parse_args()
OUT = args.output_dir.resolve()
OUT.mkdir(parents=True, exist_ok=True)
app = AppLauncher(args).app
report = {"environment_created": False, "physics_steps": 0, "tasks": []}
try:
    import gymnasium as gym
    import torch
    import ant
    dimensions = {"base_lin_vel": 3, "base_ang_vel": 3, "base_yaw_roll": 2,
                  "base_angle_to_target": 1, "base_up_proj": 1, "base_heading_proj": 1,
                  "joint_pos_norm": 8, "joint_vel_rel": 8, "feet_body_forces": 24, "actions": 8,
                  "height_scan": 63, "foot_contacts": 4}
    variants = [
        ("Ant-rl-v0", 59, None),
        ("Ant-rl-Ablation-Baseline-v0", 59, None),
        ("Ant-rl-Ablation-HeightScan-v0", 122, "heightscan_4096x32x1000"),
        ("Ant-rl-Ablation-HeightScan-Contact-Stock-v0", 126, "heightscan_contact_stock_4096x32x1000"),
        ("Ant-rl-Ablation-HeightScan-Contact-ModifiedReward-v0", 126, "heightscan_contact_modified_4096x32x1000"),
    ]
    def load(entry):
        module, symbol = entry.split(":")
        imported = importlib.import_module(module)
        assert Path(imported.__file__).is_relative_to(ROOT)
        return getattr(imported, symbol)()
    for task, expected, folder in variants:
        spec = gym.spec(task)
        cfg = load(spec.kwargs["env_cfg_entry_point"])
        agent = load(spec.kwargs["rsl_rl_cfg_entry_point"])
        policy = cfg.observations.policy.to_dict()
        terms = [name for name, value in policy.items() if isinstance(value, dict) and "func" in value]
        assert sum(dimensions[name] for name in terms) == expected
        groups = [name for name, value in vars(cfg.observations).items() if value is not None]
        assert groups == ["policy"], groups
        assert agent.obs_groups == {"policy": ["policy"], "critic": ["policy"]}
        assert agent.policy.class_name == "ActorCritic"
        assert agent.policy.actor_hidden_dims == [400, 200, 100]
        assert agent.policy.critic_hidden_dims == [400, 200, 100]
        # Compare retained terms and PPO against saved training configs, excluding runtime-only fields.
        record = {"task": task, "config_import": "PASS", "observation_terms": terms,
                  "expected_input_dim_from_terms": expected, "runner_class": agent.policy.class_name,
                  "default_num_envs": cfg.scene.num_envs}
        if folder:
            base = ROOT / "experiments/observation_ablation" / folder
            saved = json.loads((base / "results/config.json").read_text())
            assert set(saved["observations"]["policy"]) == set(policy)
            def simplify(value):
                if isinstance(value, slice): return {"slice": [value.start, value.stop, value.step]}
                if callable(value): return value.__module__ + ":" + value.__name__
                if isinstance(value, dict): return {k: simplify(v) for k, v in value.items()}
                if isinstance(value, (tuple, list)): return [simplify(v) for v in value]
                return value
            def differences(a, b, path=""):
                if isinstance(a, dict) and isinstance(b, dict):
                    return [item for key in a.keys() | b.keys() for item in differences(a.get(key), b.get(key), path + "/" + key)]
                return [] if a == b else [{"path": path, "current": str(a), "saved": str(b)}]
            report["policy_differences"] = differences(simplify(policy), saved["observations"]["policy"])
            assert not report["policy_differences"], report["policy_differences"]
            assert simplify(cfg.scene.height_scanner.to_dict()) == saved["scene"]["height_scanner"]
            checkpoint = base / "checkpoints/best_model.pt"
            digest = hashlib.file_digest(checkpoint.open("rb"), "sha256").hexdigest()
            summary = json.loads((base / "training_summary.json").read_text())
            assert digest == summary["checkpoint_sha256"]
            data = torch.load(checkpoint, map_location="cpu", weights_only=True)
            state = data["model_state_dict"]
            assert state["actor.0.weight"].shape == (400, expected)
            assert state["critic.0.weight"].shape == (400, expected)
            assert state["actor.6.weight"].shape == (8, 100)
            assert all(torch.isfinite(t).all() for t in state.values())
            record.update(checkpoint_sha256=digest, checkpoint_input_dim=expected,
                          saved_policy_and_scanner_parity="PASS", saved_iteration=data["iter"])
        report["tasks"].append(record)
    report["status"] = "PASS"
except BaseException:
    report["status"] = "FAIL"
    report["traceback"] = traceback.format_exc()
with (OUT / "runtime_smoke.json").open("x") as f:
    json.dump(report, f, indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())
print(json.dumps(report, indent=2), flush=True)
# Previous reports document unstable Kit teardown. Flush evidence, request normal shutdown.
app.close()
sys.exit(0 if report["status"] == "PASS" else 1)
