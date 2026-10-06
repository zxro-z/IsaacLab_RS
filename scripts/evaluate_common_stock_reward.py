"""Evaluation only: selected policies retain their inputs and share exact Stock Reward.

Run verify, all loaded-policy smoke tests, then evaluate (see result README).
Canonical evaluator aggregation semantics are preserved; canonical scripts are
CLI entry points with import-time side effects, so they are not imported here.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "experiments/observation_ablation/common_stock_reward_evaluation"
sys.path.insert(0, str(ROOT / "source"))
for package in ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_rl"):
    sys.path.insert(0, str(ROOT / "source" / package))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("mode", choices=["verify", "smoke", "evaluate"])
parser.add_argument("--policy", choices=["baseline", "heightscan_stock", "heightscan_contact_stock", "heightscan_contact_modified_trained"])
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--verification-record", type=Path, default=RESULT / "reward_config_verification.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
from pxr import UsdGeom
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import ant
from ant.ablation_env_cfg import AblationBaseCfg, AblationHeightScanCfg
from ant.contact_stock_env_cfg import HeightScanContactStockCfg
from ant.stage2_env_cfg import Stage2HeightScanContactCfg
from ant.agents.ablation_budget_ppo_cfg import AblationBaselinePPORunnerCfg, AblationBudgetPPORunnerCfg
from ant.agents.contact_stock_ppo_cfg import HeightScanContactStockPPORunnerCfg
from ant.agents.stage2_ppo_cfg import Stage2HeightScanContactPPORunnerCfg
from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import RewardsCfg

POLICIES = {
    "heightscan_stock": ("Ant-rl-Ablation-HeightScan-v0", AblationHeightScanCfg, AblationBudgetPPORunnerCfg, 122, "Stock", "heightscan_4096x32x1000"),
    "heightscan_contact_stock": ("Ant-rl-Ablation-HeightScan-Contact-Stock-v0", HeightScanContactStockCfg, HeightScanContactStockPPORunnerCfg, 126, "Stock", "heightscan_contact_stock_4096x32x1000"),
    "heightscan_contact_modified_trained": ("Ant-rl-Ablation-HeightScan-Contact-ModifiedReward-v0", Stage2HeightScanContactCfg, Stage2HeightScanContactPPORunnerCfg, 126, "Modified", "heightscan_contact_modified_4096x32x1000"),
    "baseline": ("Ant-rl-Ablation-Baseline-v0", AblationBaseCfg, AblationBaselinePPORunnerCfg, 59, "Stock", "baseline_4096x32x1000"),
}
TERMS = ["progress", "alive", "upright", "move_to_target", "action_l2", "energy", "joint_pos_limits"]
WEIGHTS = [1.0, 0.5, 0.1, 0.5, -0.005, -0.05, -0.1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def serialize(value):
    if isinstance(value, slice):
        return {"slice": [value.start, value.stop, value.step]}
    if isinstance(value, Path):
        return str(value)
    if callable(value):
        return value.__module__ + "." + value.__qualname__
    raise TypeError(type(value))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=serialize, allow_nan=False).encode()).hexdigest()


def dump(directory, name, value):
    path = directory / name
    assert not path.exists(), path
    path.write_text(json.dumps(value, indent=2, default=serialize, allow_nan=False) + "\n")


def stats(values):
    array = np.asarray(values, dtype=np.float64)
    assert np.isfinite(array).all()
    return dict(mean=float(array.mean()), std=float(array.std(ddof=0)), min=float(array.min()), max=float(array.max()))


def configuration(key, n=100):
    task, cls, runner_cls, dim, training_reward, folder = POLICIES[key]
    cfg, agent = cls(), runner_cls()
    cfg.seed, cfg.scene.num_envs = 24, n
    if args.device:
        cfg.sim.device = agent.device = args.device
    before = cfg.to_dict()
    before.pop("rewards")
    original_reward = cfg.rewards.to_dict()
    cfg.rewards = RewardsCfg()  # Reuse the exact stock class, not a reconstructed reward.
    after = cfg.to_dict()
    after.pop("rewards")
    assert before == after, "Reward replacement changed non-reward configuration"
    assert cfg.rewards.to_dict() == RewardsCfg().to_dict()
    checkpoint = ROOT / "experiments/observation_ablation" / folder / "checkpoints/best_model.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    weights = saved["model_state_dict"]
    assert weights["actor.0.weight"].shape[1] == weights["critic.0.weight"].shape[1] == dim
    assert agent.policy.actor_obs_normalization is False and agent.policy.critic_obs_normalization is False
    record = dict(policy=key, task=task, checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=sha(checkpoint),
                  selected_iteration=saved["iter"], observation_dimension=dim, training_reward=training_reward,
                  original_training_reward_config=original_reward, evaluation_reward="Stock", active_evaluation_reward_config=cfg.rewards.to_dict(),
                  evaluation_seed=24, terrain_seed=cfg.scene.terrain.terrain_generator.seed, num_envs=n,
                  episode_length_s=cfg.episode_length_s, max_control_steps=960, control_dt=cfg.sim.dt * cfg.decimation,
                  termination_config=cfg.terminations.to_dict(), reset_events=cfg.events.to_dict(),
                  reward_only_replacement=True, observation_config_sha256=digest(cfg.observations.to_dict()))
    return cfg, agent, record


def verify():
    configs, records = [], []
    for key in POLICIES:
        cfg, agent, record = configuration(key)
        assert gym.spec(record["task"]).kwargs["env_cfg_entry_point"].endswith(type(cfg).__name__)
        configs.append(cfg.to_dict())
        records.append(record)
    assert configs[1] == configs[2], "Contact configs differ beyond original training reward"
    reference = configs[0]
    contact = configs[1]
    contact["scene"].pop("feet_contacts")
    contact["observations"]["policy"].pop("foot_contacts")
    assert reference == contact, "Environment configs differ beyond intended Contact interface/sensor"
    import copy
    without_scan = copy.deepcopy(reference)
    without_scan["scene"].pop("height_scanner")
    without_scan["observations"]["policy"].pop("height_scan")
    assert without_scan == configs[3], "Baseline differs beyond missing HeightScan interface/sensor"
    assert records[0]["terrain_seed"] == 42
    assert records[0]["episode_length_s"] == 16 and records[0]["control_dt"] == 1 / 60
    args.output.mkdir(parents=True, exist_ok=True)
    dump(args.output, "reward_config_verification.json", dict(pass_=True, policies=records,
         contact_configs_identical_after_reward_replacement=True, common_environment_identical=True,
         baseline_config_identical_except_heightscan=True,
         allowed_difference="Baseline lacks HeightScan sensor/63-D term; HeightScan-only lacks observation-only feet_contacts sensor/4-D term",
         normalization="Only evaluation seed=24 and num_envs=100; native termination/reset/terrain unchanged",
         deterministic_mean_actions=True, first_episode_only=True, terminal_reward_included=True, post_reset_reward_excluded=True))
    print("CONFIG VERIFICATION PASS: 59/122/126/126; identical Stock Reward and environment protocol", flush=True)


def rollout():
    assert args.policy is not None
    gate = json.loads(args.verification_record.read_text())
    assert gate["pass_"]
    if args.mode == "evaluate":
        for key in POLICIES:
            assert json.loads((RESULT / "smoke" / key / "smoke.json").read_text())["pass"]
    assert not args.output.exists(), args.output
    args.output.mkdir(parents=True)
    n = 4 if args.mode == "smoke" else 100
    cfg, agent, record = configuration(args.policy, n)
    verified = next(r for r in gate["policies"] if r["policy"] == args.policy)
    for key in ("checkpoint_sha256", "observation_config_sha256", "active_evaluation_reward_config"):
        assert digest(record[key]) == digest(verified[key]), key
    dump(args.output, "config.json", cfg.to_dict())
    dump(args.output, "agent.json", agent.to_dict())
    raw = gym.make(record["task"], cfg=cfg)
    base = raw.unwrapped
    env = RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)
    obs = env.get_observations()
    robot, dt = base.scene["robot"], base.step_dt
    assert obs["policy"].shape == (n, record["observation_dimension"])
    assert env.num_actions == 8 and dt == 1 / 60 and base.max_episode_length == 960
    if record["observation_dimension"] == 59:
        assert "height_scanner" not in base.scene.sensors and "feet_contacts" not in base.scene.sensors
    else:
        assert base.scene["height_scanner"].num_rays == 63
    assert base.reward_manager.active_terms == TERMS
    runtime = []
    for name, weight in zip(TERMS, WEIGHTS):
        term = base.reward_manager.get_term_cfg(name)
        expected = getattr(RewardsCfg(), name)
        func = term.func if hasattr(term.func, "__qualname__") else type(term.func)
        assert term.weight == weight == expected.weight and func is expected.func and term.params == expected.params
        runtime.append(dict(term=name, weight=weight, function=serialize(func), params=term.params))
    mesh = UsdGeom.Mesh(base.scene.stage.GetPrimAtPath("/World/ground/terrain/mesh"))
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
    mesh_hash = hashlib.sha256(points.tobytes() + np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32).tobytes()).hexdigest()
    bounds = [float(points[:, 0].min()), float(points[:, 0].max())]
    initial = dict(root_state_w=robot.data.root_state_w.cpu().tolist(), joint_pos=robot.data.joint_pos.cpu().tolist(),
                   joint_vel=robot.data.joint_vel.cpu().tolist(), env_origins=base.scene.env_origins.cpu().tolist(),
                   terrain_rows=base.scene.terrain.terrain_levels.cpu().tolist(), terrain_columns=base.scene.terrain.terrain_types.cpu().tolist(),
                   material_properties=robot.root_physx_view.get_material_properties().cpu().tolist())
    dump(args.output, "initial_conditions.json", initial)
    record.update(timestamp_utc=datetime.now(timezone.utc).isoformat(), git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  working_tree_status=subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True), clean_committed_tree=False,
                  working_tree_diff_sha256=hashlib.sha256(subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT)).hexdigest(),
                  python_version=sys.version, isaaclab_version=(ROOT / "VERSION").read_text().strip(),
                  isaacsim_version=importlib.metadata.version("isaacsim"), pytorch_version=torch.__version__, cuda_version=torch.version.cuda,
                  runtime_reward_terms=runtime, terrain_mesh_sha256=mesh_hash, terrain_x_bounds=bounds, initial_conditions_sha256=digest(initial),
                  deterministic_mean_actions=True, mode=args.mode, reward_contribution="RewardManager._step_reward * step_dt")
    sources = [Path(__file__), ROOT / "source/ant/ablation_env_cfg.py", ROOT / "source/ant/contact_stock_env_cfg.py", ROOT / "source/ant/stage2_env_cfg.py",
               ROOT / "source/ant/ant_env_cfg.py", ROOT / "source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/ant_env_cfg.py",
               ROOT / "source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/humanoid/mdp/rewards.py",
               ROOT / "source/isaaclab/isaaclab/managers/reward_manager.py", ROOT / "source/isaaclab/isaaclab/envs/manager_based_rl_env.py"]
    record["source_sha256"] = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    dump(args.output, "manifest.json", record)
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=agent.device)
    runner.load(str(ROOT / record["checkpoint"]), load_optimizer=False)
    model = runner.get_inference_policy(device=base.device)
    with torch.inference_mode():
        assert torch.equal(model(obs), model(obs)), "Inference must use deterministic mean action"
    done = torch.zeros(n, dtype=torch.bool, device=base.device)
    returns = torch.zeros(n, dtype=torch.float64, device=base.device)
    components = torch.zeros((n, len(TERMS)), dtype=torch.float64, device=base.device)
    steps = torch.zeros(n, dtype=torch.long, device=base.device)
    velocity, end, terminal_x, terminal_v = [returns.clone() for _ in range(4)]
    fall, timeout = done.clone(), done.clone()
    start = robot.data.root_pos_w[:, 0].clone()
    original_reset = base._reset_idx

    def capture_terminal(self, ids):
        terminal_x[ids] = (robot.data.root_pos_w[ids, 0] - start[ids]).double()
        terminal_v[ids] = robot.data.root_lin_vel_w[ids, 0].double()
        return original_reset(ids)

    base._reset_idx = types.MethodType(capture_terminal, base)
    max_error = 0.0
    limit = 32 if args.mode == "smoke" else base.max_episode_length
    for step in range(limit):
        assert app.is_running()
        with torch.inference_mode():
            assert obs["policy"].shape == (n, record["observation_dimension"]) and torch.isfinite(obs["policy"]).all()
            action = model(obs)
            assert action.shape == (n, 8) and torch.isfinite(action).all()
            obs, reward, dones, _ = env.step(action)
            contribution = base.reward_manager._step_reward * dt
            assert torch.isfinite(reward).all() and torch.isfinite(contribution).all() and torch.isfinite(obs["policy"]).all()
            active, new = ~done, ~done & dones.bool()
            if active.any():
                max_error = max(max_error, float((contribution.double().sum(-1) - reward.double())[active].abs().max()))
            returns[active] += reward[active].double()
            components[active] += contribution[active].double()
            steps[active] += 1
            velocity[active] += torch.where(dones.bool(), terminal_v, robot.data.root_lin_vel_w[:, 0])[active].double()
            end[new] = terminal_x[new]
            fall[new] = base.termination_manager.get_term("body_z_down")[new]
            timeout[new] = base.termination_manager.get_term("time_out")[new]
            done |= dones.bool()
        if done.all():
            break
        if (step + 1) % 120 == 0:
            print(f"[PROGRESS] {step+1}: {int(done.sum())}/{n} first episodes complete", flush=True)
    residual = (components.sum(-1) - returns).abs()
    assert max_error < 1e-5 and float(residual.max()) < 1e-3
    assert sha(ROOT / record["checkpoint"]) == record["checkpoint_sha256"]
    if args.mode == "smoke":
        dump(args.output, "smoke.json", dict(**{"pass": True}, policy=args.policy, checkpoint_loaded=True, actor_forward=True,
             deterministic=True, observation_dimension=record["observation_dimension"], finite=True, num_envs=n, steps=step+1,
             evaluation_reward="Stock", runtime_reward_terms=runtime, max_step_reward_residual=max_error, metrics_are_experiment_results=False))
    else:
        assert done.all() and n == 100
        rows = []
        for i in range(n):
            row = dict(env_id=i, episode_return=float(returns[i]), episode_steps=int(steps[i]), episode_duration=int(steps[i])*dt,
                       forward_displacement=float(end[i]), mean_forward_velocity=float(velocity[i]/steps[i]), fall=bool(fall[i]), timeout=bool(timeout[i]),
                       other=not bool(fall[i] | timeout[i]), initial_world_x=float(start[i]), terminal_world_x=float(start[i]+end[i]),
                       out_of_terrain_x=bool(start[i]+end[i] < bounds[0] or start[i]+end[i] > bounds[1]), reward_identity_residual=float(residual[i]),
                       terrain_row=initial["terrain_rows"][i], terrain_column=initial["terrain_columns"][i])
            row.update({"reward_" + name: float(components[i, j]) for j, name in enumerate(TERMS)})
            rows.append(row)
        with (args.output / "episodes.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        metrics = {key: stats([r[key] for r in rows]) for key in ("episode_return", "forward_displacement", "episode_duration", "mean_forward_velocity")}
        for key in ("fall", "timeout", "other", "out_of_terrain_x"):
            count = sum(r[key] for r in rows)
            metrics[key] = dict(count=count, ratio=count/n)
        for distance in (2, 5, 10):
            count = sum(r["forward_displacement"] >= distance for r in rows)
            metrics[f">={distance}m"] = dict(count=count, ratio=count/n)
        dump(args.output, "summary.json", dict(metrics=metrics, reward_components={name: stats(components[:, j].cpu().numpy()) for j, name in enumerate(TERMS)},
             completed_first_episodes=int(done.sum()), finite=True, first_episode_only=True, terminal_step_included=True, post_reset_rewards_excluded=True,
             max_step_reward_residual=max_error, max_episode_reward_residual=float(residual.max()), evaluation_reward="Stock"))
        print(json.dumps(metrics), flush=True)
    env.close()


try:
    verify() if args.mode == "verify" else rollout()
except BaseException:
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)  # Match existing evaluators: avoid Kit shutdown hanging on exceptions.
else:
    app.close()
