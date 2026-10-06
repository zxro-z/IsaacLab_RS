"""Frozen HeightScan smoke, training and first-episode evaluation.

Run with the existing IsaacLab_RS Python environment; see experiment README.
Each output directory must be new. No historical artifacts are written.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
from submission_provenance import manifest_git_provenance
import sys
import types
import time

ROOT = Path(__file__).resolve().parents[1]
RS = Path(os.environ.get("ISAACLAB_RS", "/home/zxro/IsaacLab_RS"))
sys.path.insert(0, str(ROOT / "source"))
if os.environ.get("ISAAC_SIM_SITE_PACKAGES"):
    sys.path.append(os.environ["ISAAC_SIM_SITE_PACKAGES"])
for package in ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_rl"):
    sys.path.insert(0, str(RS / "source" / package))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=["smoke", "train", "evaluate"])
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
args.output.mkdir(parents=True)
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
from pxr import UsdGeom
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import ant
from ant.ant_env_cfg import AntEnvCfg
from ant.ablation_env_cfg import AblationHeightScanCfg
from ant.agents.ablation_ppo_cfg import AblationHeightScanPPORunnerCfg
from ant.agents.ablation_budget_ppo_cfg import AblationBudgetPPORunnerCfg
from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import RewardsCfg
from isaaclab_tasks.manager_based.classic.ant.ant_terrain_heightscan_env_cfg import height_scanner_cfg

TASK = "Ant-rl-Ablation-HeightScan-v0"
TERMS = ["progress", "alive", "upright", "move_to_target", "action_l2", "energy", "joint_pos_limits"]
WEIGHTS = [1.0, 0.5, 0.1, 0.5, -0.005, -0.05, -0.1]
SELECTION = "best_model: highest training logged completed-episode mean return; no evaluation-based selection"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def serialize(value):
    if isinstance(value, slice):
        return {"slice": [value.start, value.stop, value.step]}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if callable(value):
        return value.__module__ + "." + value.__qualname__
    raise TypeError(type(value))


def dump(name, value):
    (args.output / name).write_text(json.dumps(value, indent=2, default=serialize, allow_nan=False) + "\n")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=serialize).encode()).hexdigest()


def stats(value):
    a = np.asarray(value, dtype=np.float64)
    assert np.isfinite(a).all()
    return dict(mean=float(a.mean()), std=float(a.std(ddof=0)), min=float(a.min()), max=float(a.max()))


def write_csv(name, rows):
    with (args.output / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


class BestModelRunner(OnPolicyRunner):
    """Same preregistered selection statistic as historical Team1 training."""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.best_mean_reward = float("-inf")

    def log(self, locs, width=80, pad=35):
        super().log(locs, width, pad)
        if locs["rewbuffer"]:
            mean = statistics.mean(locs["rewbuffer"])
            if mean > self.best_mean_reward:
                self.best_mean_reward = mean
                self.save(str(Path(self.log_dir) / "best_model.pt"), infos={"mean_reward": mean})


def main():
    cfg = AblationHeightScanCfg()
    agent = AblationBudgetPPORunnerCfg()
    historical = AblationHeightScanPPORunnerCfg()
    assert agent.policy.to_dict() == historical.policy.to_dict()
    assert agent.algorithm.to_dict() == historical.algorithm.to_dict()
    assert agent.num_steps_per_env == 32 and agent.max_iterations == 1000
    assert not agent.resume and agent.load_run is None and agent.load_checkpoint is None
    original = AntEnvCfg()
    frozen = {"terrain": cfg.scene.terrain.to_dict(), "robot": cfg.scene.robot.to_dict(),
              "sim": cfg.sim.to_dict(), "actions": cfg.actions.to_dict(),
              "events": cfg.events.to_dict(), "terminations": cfg.terminations.to_dict(),
              "episode_length_s": cfg.episode_length_s, "decimation": cfg.decimation}
    for key in ("terrain", "robot"):
        assert frozen[key] == getattr(original.scene, key).to_dict(), key
    for key in ("sim", "actions", "events", "terminations"):
        assert frozen[key] == getattr(original, key).to_dict(), key
    assert cfg.episode_length_s == original.episode_length_s and cfg.decimation == original.decimation
    assert cfg.rewards.to_dict() == RewardsCfg().to_dict()
    assert cfg.scene.height_scanner.to_dict() == height_scanner_cfg().to_dict()
    assert agent.algorithm.to_dict() == ant.agents.rsl_rl_ppo_cfg.AntPPORunnerCfg().algorithm.to_dict()
    # Preserve every native proprio term, including existing incoming wrench.
    policy = cfg.observations.policy.to_dict()
    scan_term = policy.pop("height_scan")
    assert policy == original.observations.policy.to_dict()
    assert not any("contact" in name.lower() for name in policy)
    cfg.seed = 42 if args.mode == "train" else 24
    cfg.scene.num_envs = {"smoke": 4, "train": 4096, "evaluate": 100}[args.mode]
    if args.device:
        cfg.sim.device = args.device
        agent.device = args.device
    cfg.log_dir = str(args.output)
    dump("source_parity.json", {"frozen": frozen, "hashes": {k: digest(v) for k, v in frozen.items()},
                              "reward": cfg.rewards.to_dict(), "reward_hash": digest(cfg.rewards.to_dict()),
                              "ppo": agent.algorithm.to_dict(), "ppo_hash": digest(agent.algorithm.to_dict()),
                              "native_proprio_preserved": True, "canonical_scan_preserved": True})
    dump("config.json", cfg.to_dict())
    dump("agent.json", agent.to_dict())
    sources = [Path(__file__), ROOT / "source/ant/ablation_env_cfg.py",
               ROOT / "source/ant/agents/ablation_ppo_cfg.py", ROOT / "source/ant/agents/ablation_budget_ppo_cfg.py", ROOT / "source/ant/ant_env_cfg.py",
               ROOT / "source/ant/agents/rsl_rl_ppo_cfg.py", ROOT / "source/ant/__init__.py",
               RS / "source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/ant_env_cfg.py",
               RS / "source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/ant_terrain_heightscan_env_cfg.py",
               RS / "source/isaaclab/isaaclab/managers/reward_manager.py",
               RS / "source/isaaclab/isaaclab/envs/mdp/observations.py",
               RS / "source/isaaclab/isaaclab/sensors/ray_caster/ray_caster.py",
               RS / "source/isaaclab/isaaclab/sensors/ray_caster/patterns/patterns.py"]
    dump("source_mapping.json", {str(p): sha(p) for p in sources})
    env = gym.make(TASK, cfg=cfg)
    base = env.unwrapped
    env = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    obs = env.get_observations()
    n = env.num_envs
    dt = base.step_dt
    robot = base.scene["robot"]
    assert obs["policy"].shape == (n, 122)
    assert list(obs.keys()) == ["policy"]
    assert env.num_actions == 8 and abs(dt - 1 / 60) < 1e-12
    assert base.reward_manager.active_terms == TERMS
    print(base.reward_manager, flush=True)
    runtime = []
    for term, weight in zip(TERMS, WEIGHTS):
        t = base.reward_manager.get_term_cfg(term)
        assert t.weight == weight
        f = t.func if hasattr(t.func, "__qualname__") else type(t.func)
        runtime.append(dict(term=term, weight=t.weight, function=f.__module__ + "." + f.__qualname__, params=t.params))
    sensor = base.scene["height_scanner"]
    assert sensor.num_rays == 63
    scan = obs["policy"][:, 59:]
    assert torch.isfinite(obs["policy"]).all()
    assert torch.isfinite(sensor.data.ray_hits_w).all()
    # Remove per-robot mean to show terrain geometry rather than only torso height.
    centered = scan - scan.mean(-1, keepdim=True)
    variation = float(centered.std(dim=0).max())
    assert variation > 1e-5, "No geometric scan variation across terrain locations"
    samples = [dict(env_id=i, terrain_row=int(base.scene.terrain.terrain_levels[i]),
                    terrain_column=int(base.scene.terrain.terrain_types[i]),
                    origin=base.scene.env_origins[i].cpu().tolist(),
                    scan_min=float(scan[i].min()), scan_max=float(scan[i].max()), scan_mean=float(scan[i].mean()))
               for i in range(min(n, 8))]
    print(json.dumps(samples), flush=True)
    mesh = UsdGeom.Mesh(base.scene.stage.GetPrimAtPath("/World/ground/terrain/mesh"))
    mesh_points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
    terrain_x_bounds = [float(mesh_points[:, 0].min()), float(mesh_points[:, 0].max())]
    mesh_sha = hashlib.sha256(np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32).tobytes()
                             + np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32).tobytes()).hexdigest()
    manifest = dict(task=TASK, **manifest_git_provenance(ROOT, allow_snapshot=args.mode == "evaluate"),
                    training_environment="Team1 AntEnvCfg", evaluation_environment="Team1 AntEnvCfg (same dynamics/terrain)",
                    training_seed=42, evaluation_seed=24, terrain_seed=42,
                    num_envs=cfg.scene.num_envs, training_num_envs=4096, evaluation_num_envs=100,
                    iterations=1000, num_steps_per_env=32, total_transitions=131072000, resume=False, load_run=None, load_checkpoint=None, checkpoint_interval=50, checkpoint_selection=SELECTION,
                    control_dt=dt, physics_dt=cfg.sim.dt, episode_length_s=16.0,
                    observation=dict(proprio=59, heightscan=63, actor=122, critic=122,
                                     order=list(base.observation_manager.active_terms["policy"]),
                                     scan_formula="sensor.data.pos_w[:,2] - ray_hits_w[...,2] - 0.5",
                                     clip=[-1, 1], scale=1.0, empirical_normalization=False,
                                     grid=[9, 7], ordering="xy (x varies fastest)",
                                     offset=[0.8, 0, 20], direction=[0, 0, -1], max_distance=1e6,
                                     added_contact_observation=False, cnn=False,
                                     native_foot_wrench_preserved=True),
                    architecture=agent.policy.to_dict(), ppo=agent.algorithm.to_dict(), reward=runtime,
                    terrain_mesh_sha256=mesh_sha, terrain_x_bounds=terrain_x_bounds, terrain_samples=samples,
                    evaluation_protocol="100 envs, seed24, deterministic mean actions, first episode only, terminal step included",
                    fall_semantics="body_z_down: bad_orientation(pi/2), no torso height termination",
                    reward_contribution="RewardManager._step_reward * actual step_dt = raw * weight * dt")
    dump("manifest.json", manifest)
    if args.mode == "train":
        smoke = ROOT / "experiments/observation_ablation/heightscan_4096x32x1000/smoke_verified/smoke.json"
        assert json.loads(smoke.read_text())["pass"], "Smoke must PASS before training"
        # Training has exactly the same core source hashes as the validated smoke.
        assert json.loads((smoke.parent / "source_mapping.json").read_text()) == json.loads((args.output / "source_mapping.json").read_text())
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
        runner = BestModelRunner(env, agent.to_dict(), log_dir=str(args.output), device=agent.device)
        started = time.perf_counter()
        print("FRESH TRAINING: resume=False, 4096 envs, 32 steps, 1000 iterations", flush=True)
        runner.learn(num_learning_iterations=1000, init_at_random_ep_len=True)
        wall_time = time.perf_counter() - started
        assert runner.current_learning_iteration == 999
        assert (args.output / "model_999.pt").is_file()
        selected = args.output / "best_model.pt"
        checkpoint = torch.load(selected, map_location="cpu", weights_only=False)
        dump("training_summary.json", dict(completed=True, iterations=1000, num_steps_per_env=32, total_transitions=131072000, resume=False, load_run=None, load_checkpoint=None, selected_checkpoint=str(selected),
             wall_time_seconds=wall_time, final_checkpoint=str(args.output / "model_999.pt"), final_checkpoint_sha256=sha(args.output / "model_999.pt"), checkpoint_bytes=selected.stat().st_size, checkpoint_sha256=sha(selected), selected_iteration=checkpoint["iter"], selection=SELECTION,
             selected_training_mean_reward=checkpoint.get("infos", {}).get("mean_reward")))
        env.close()
        return

    model = None
    if args.mode == "evaluate":
        assert args.checkpoint is not None
        runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=agent.device)
        runner.load(str(args.checkpoint))
        model = runner.get_inference_policy(device=base.device)
        manifest["checkpoint"] = str(args.checkpoint)
        manifest["checkpoint_sha256"] = sha(args.checkpoint)
        dump("manifest.json", manifest)
    done = torch.zeros(n, dtype=torch.bool, device=base.device)
    returns = torch.zeros(n, dtype=torch.float64, device=base.device)
    components = torch.zeros((n, 7), dtype=torch.float64, device=base.device)
    steps = torch.zeros(n, dtype=torch.long, device=base.device)
    velocity = returns.clone()
    start = robot.data.root_pos_w[:, 0].clone()
    end = returns.clone()
    terminal_x = returns.clone()
    terminal_v = returns.clone()
    fall = done.clone()
    timeout = done.clone()
    original_reset = base._reset_idx

    def capture_terminal(self, ids):
        terminal_x[ids] = (robot.data.root_pos_w[ids, 0] - start[ids]).double()
        terminal_v[ids] = robot.data.root_lin_vel_w[ids, 0].double()
        return original_reset(ids)

    base._reset_idx = types.MethodType(capture_terminal, base)
    max_error = 0.0
    previous_scan = scan.clone()
    changed = False
    missing_hit_samples = 0
    limit = 32 if args.mode == "smoke" else base.max_episode_length
    for step in range(limit):
        assert app.is_running()
        with torch.inference_mode():
            assert obs["policy"].shape == (n, 122) and torch.isfinite(obs["policy"]).all()
            missing_hit_samples += int(torch.isinf(sensor.data.ray_hits_w[~done]).any(-1).sum())
            action = model(obs) if model else torch.zeros((n, 8), device=base.device)
            assert action.shape == (n, 8) and torch.isfinite(action).all()
            obs, reward, dones, _ = env.step(action)
            assert torch.isfinite(reward).all() and torch.isfinite(obs["policy"]).all()
            if args.mode == "smoke":
                assert torch.isfinite(sensor.data.ray_hits_w).all()
            else:
                # Native RayCaster uses Inf for a missed static-mesh ray. The
                # canonical height_scan clip maps it to -1; do not change the
                # terrain or the representation to satisfy a raw-data check.
                assert not torch.isnan(sensor.data.ray_hits_w).any()
            contribution = base.reward_manager._step_reward * dt
            assert torch.isfinite(contribution).all()
            active = ~done
            new = active & dones.bool()
            if active.any():
                max_error = max(max_error, float((contribution.double().sum(-1) - reward.double())[active].abs().max()))
            returns[active] += reward[active].double()
            components[active] += contribution[active].double()
            steps[active] += 1
            velocity[active] += torch.where(dones.bool(), terminal_v, robot.data.root_lin_vel_w[:, 0])[active].double()
            end[new] = terminal_x[new]
            fall[new] = base.termination_manager.get_term("body_z_down")[new]
            timeout[new] = base.termination_manager.get_term("time_out")[new]
            changed |= bool((obs["policy"][:, 59:] - previous_scan).abs().max() > 1e-5)
            done |= dones.bool()
        if done.all():
            break
        if (step + 1) % 120 == 0:
            print(f"[PROGRESS] {step+1}: {int(done.sum())}/{n} first episodes complete", flush=True)
    residual = (components.sum(-1) - returns).abs()
    assert max_error < 1e-5 and float(residual.max()) < 1e-3
    if args.mode == "smoke":
        assert changed
        dump("smoke.json", dict(pass_=True, **{"pass": True}, registration=True, terrain=True, robot=True,
             raycaster=True, proprio=59, heightscan=63, actor=122, actions=8, finite=True,
             reward_terms=runtime, custom_reward_active=False, added_contact_observation=False,
             geometric_scan_variation=variation, scan_changed_during_steps=changed, steps=step+1,
             max_step_reward_residual=max_error, samples=samples))
    else:
        assert done.all(), "All first episodes must finish"
        other = ~(fall | timeout)
        rows = []
        for i in range(n):
            row = dict(env_id=i, episode_return=float(returns[i]), episode_steps=int(steps[i]),
                       episode_duration=int(steps[i])*dt, forward_displacement=float(end[i]),
                       mean_forward_velocity=float(velocity[i]/steps[i]), fall=bool(fall[i]), timeout=bool(timeout[i]),
                       other=bool(other[i]), initial_world_x=float(start[i]), terminal_world_x=float(start[i]+end[i]), out_of_terrain_x=bool(start[i]+end[i] < terrain_x_bounds[0] or start[i]+end[i] > terrain_x_bounds[1]), reward_identity_residual=float(residual[i]),
                       terrain_row=int(base.scene.terrain.terrain_levels[i]), terrain_column=int(base.scene.terrain.terrain_types[i]))
            row.update({"reward_"+name:float(components[i,j]) for j,name in enumerate(TERMS)})
            rows.append(row)
        write_csv("episode_metrics.csv", rows)
        metrics = {key:stats([r[key] for r in rows]) for key in
                   ["episode_return", "episode_steps", "episode_duration", "forward_displacement", "mean_forward_velocity"]}
        metrics.update({key:dict(count=int(value.sum()), ratio=float(value.float().mean()))
                        for key,value in [("fall",fall),("timeout",timeout),("other",other)]})
        for distance in [2, 5, 10]:
            metrics[f">={distance}m"] = dict(count=int((end>=distance).sum()), ratio=float((end>=distance).float().mean()))
        metrics["out_of_terrain_x"] = dict(count=sum(r["out_of_terrain_x"] for r in rows), ratio=sum(r["out_of_terrain_x"] for r in rows)/n)
        decomposition = {name:stats(components[:,j].cpu().numpy()) for j,name in enumerate(TERMS)}
        decomposition["total"] = stats(returns.cpu().numpy())
        write_csv("main_metrics.csv", [dict(metric=k, **v) for k,v in metrics.items() if "mean" in v])
        write_csv("reward_components.csv", [dict(component=k, **v) for k,v in decomposition.items()])
        dump("evaluation_summary.json", dict(metrics=metrics, reward_components=decomposition,
             max_step_reward_residual=max_error, max_episode_reward_residual=float(residual.max()), mean_episode_reward_residual=float(residual.mean()), terrain_x_bounds=terrain_x_bounds,
             finite=True, first_episode_only=True, terminal_step_included=True, post_reset_rewards_excluded=True,
             terrain_mesh_sha256=mesh_sha, simultaneous_fall_timeout_count=int((fall&timeout).sum())))
        dump("raycast_diagnostics.json", dict(raw_missing_ray_samples=missing_hit_samples,
             interpretation="Inf raw hit = no mesh intersection; canonical height_scan clip maps to -1",
             processed_observations_finite=True, raw_nan=False))
    env.close()


try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    # Kit shutdown can hang after initialization/config exceptions; fail visibly.
    os._exit(1)
finally:
    app.close()
