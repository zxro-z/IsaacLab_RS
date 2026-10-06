"""Evaluate DepthCam checkpoints with the observation-ablation protocol."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
ISAACLAB = Path(os.environ.get("ISAACLAB_ROOT", "/home/user/package/Isaaclab-2.3.0"))
sys.path.insert(0, str(ROOT / "source"))
for package in ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_rl"):
    sys.path.insert(0, str(ISAACLAB / "source" / package))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--stage", choices=("stock", "modified"), required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
from pxr import UsdGeom
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import RewardsCfg as StockRewardsCfg

import ant
from ant import rewards as team_rewards
from ant.agents.rsl_rl_ppo_cfg import AntPPORunnerCfg
from ant.ant_env_cfg import AntEnvCfg
from ant.depth_actor_critic import register_depth_actor_critic

register_depth_actor_critic()


#########################
# Result helpers
#########################


def stats(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


#########################
# Evaluation
#########################


def main():
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    cfg = AntEnvCfg()
    agent = AntPPORunnerCfg()
    cfg.seed = 24
    cfg.scene.num_envs = 100
    cfg.log_dir = str(args.output)
    if args.device:
        cfg.sim.device = args.device
        agent.device = args.device

    if args.stage == "stock":
        cfg.rewards = StockRewardsCfg()
        terms = ["progress", "alive", "upright", "move_to_target", "action_l2", "energy", "joint_pos_limits"]
    else:
        terms = [
            "progress",
            "alive",
            "upright",
            "move_to_target",
            "foot_contact",
            "action_l2",
            "energy",
            "joint_velocity",
            "joint_pos_limits",
            "foot_slip",
        ]

    env = gym.make("Ant-rl-v0", cfg=cfg)
    base = env.unwrapped
    robot = base.scene["robot"]
    if args.stage == "stock":
        expected_weights = [1.0, 0.5, 0.1, 0.5, -0.005, -0.05, -0.1]
        assert base.reward_manager.active_terms == terms
        assert [base.reward_manager.get_term_cfg(name).weight for name in terms] == expected_weights
    else:
        expected_weights = [2.5, 0.5, 0.05, 1.5, 1.0, -0.005, -0.15, -0.001, -0.5, -0.07]
        configured_weights = [
            team_rewards.PROGRESS_WEIGHT,
            team_rewards.ALIVE_WEIGHT,
            team_rewards.UPRIGHT_WEIGHT,
            team_rewards.HEADING_WEIGHT,
            team_rewards.CONTACT_WEIGHT,
            team_rewards.ACTION_WEIGHT,
            team_rewards.ENERGY_WEIGHT,
            team_rewards.JOINT_VEL_WEIGHT,
            team_rewards.JOINT_LIMIT_WEIGHT,
            team_rewards.FOOT_SLIP_WEIGHT,
        ]
        assert base.reward_manager.active_terms == ["total_reward"]
        assert configured_weights == expected_weights
    env = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=agent.device)
    runner.load(str(args.checkpoint))
    policy = runner.get_inference_policy(device=base.device)

    observations = env.get_observations()
    assert observations["policy"].shape == (100, 59)
    assert observations["depth"].shape == (100, 48, 64, 1)

    count = env.num_envs
    dt = base.step_dt
    finished = torch.zeros(count, dtype=torch.bool, device=base.device)
    returns = torch.zeros(count, dtype=torch.float64, device=base.device)
    components = torch.zeros((count, len(terms)), dtype=torch.float64, device=base.device)
    steps = torch.zeros(count, dtype=torch.long, device=base.device)
    velocity_sum = torch.zeros(count, dtype=torch.float64, device=base.device)
    start_x = robot.data.root_pos_w[:, 0].clone()
    displacement = torch.zeros(count, dtype=torch.float64, device=base.device)
    terminal_velocity = torch.zeros(count, dtype=torch.float64, device=base.device)
    fall = torch.zeros(count, dtype=torch.bool, device=base.device)
    timeout = torch.zeros(count, dtype=torch.bool, device=base.device)

    original_reset = base._reset_idx

    def capture_terminal(self, env_ids):
        displacement[env_ids] = (robot.data.root_pos_w[env_ids, 0] - start_x[env_ids]).double()
        terminal_velocity[env_ids] = robot.data.root_lin_vel_w[env_ids, 0].double()
        return original_reset(env_ids)

    base._reset_idx = types.MethodType(capture_terminal, base)

    captured = {}
    if args.stage == "modified":
        reward_code = team_rewards.TotalReward.__call__.__code__

        def capture_reward(frame, event, value):
            if event == "return" and frame.f_code is reward_code:
                captured["weighted"] = frame.f_locals["weighted_rewards"]

    max_step_error = 0.0
    for _ in range(base.max_episode_length):
        with torch.inference_mode():
            actions = policy(observations)
            if args.stage == "modified":
                previous_profile = sys.getprofile()
                sys.setprofile(capture_reward)
                try:
                    observations, reward, dones, _ = env.step(actions)
                finally:
                    sys.setprofile(previous_profile)
                contribution = torch.stack([captured["weighted"][name] for name in terms], dim=-1) * dt
            else:
                observations, reward, dones, _ = env.step(actions)
                contribution = base.reward_manager._step_reward * dt

            active = ~finished
            newly_finished = active & dones.bool()
            max_step_error = max(
                max_step_error,
                float((contribution.double().sum(-1) - reward.double())[active].abs().max()),
            )
            returns[active] += reward[active].double()
            components[active] += contribution[active].double()
            steps[active] += 1
            step_velocity = torch.where(dones.bool(), terminal_velocity, robot.data.root_lin_vel_w[:, 0])
            velocity_sum[active] += step_velocity[active].double()
            fall[newly_finished] = base.termination_manager.get_term("body_z_down")[newly_finished]
            timeout[newly_finished] = base.termination_manager.get_term("time_out")[newly_finished]
            finished |= dones.bool()
        if finished.all():
            break

    assert finished.all()
    residual = (components.sum(-1) - returns).abs()
    assert max_step_error < 1.0e-5
    assert float(residual.max()) < 1.0e-3

    terrain = UsdGeom.Mesh(base.scene.stage.GetPrimAtPath("/World/ground/terrain/mesh"))
    points = np.asarray(terrain.GetPointsAttr().Get(), dtype=np.float32)
    faces = np.asarray(terrain.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
    terrain_x_bounds = [float(points[:, 0].min()), float(points[:, 0].max())]
    terrain_mesh_sha256 = hashlib.sha256(points.tobytes() + faces.tobytes()).hexdigest()
    duration = steps.double() * dt
    mean_velocity = velocity_sum / steps
    terminal_x = start_x.double() + displacement
    other = ~(fall | timeout)

    rows = []
    for env_id in range(count):
        row = {
            "env_id": env_id,
            "episode_return": float(returns[env_id]),
            "episode_steps": int(steps[env_id]),
            "episode_duration": float(duration[env_id]),
            "forward_displacement": float(displacement[env_id]),
            "mean_forward_velocity": float(mean_velocity[env_id]),
            "fall": bool(fall[env_id]),
            "timeout": bool(timeout[env_id]),
            "other": bool(other[env_id]),
            "initial_world_x": float(start_x[env_id]),
            "terminal_world_x": float(terminal_x[env_id]),
            "out_of_terrain_x": bool(
                terminal_x[env_id] < terrain_x_bounds[0] or terminal_x[env_id] > terrain_x_bounds[1]
            ),
            "reward_identity_residual": float(residual[env_id]),
        }
        row.update({f"reward_{name}": float(components[env_id, index]) for index, name in enumerate(terms)})
        rows.append(row)

    metrics = {
        "episode_return": stats(returns.cpu().numpy()),
        "episode_steps": stats(steps.cpu().numpy()),
        "episode_duration": stats(duration.cpu().numpy()),
        "forward_displacement": stats(displacement.cpu().numpy()),
        "mean_forward_velocity": stats(mean_velocity.cpu().numpy()),
        "fall": {"count": int(fall.sum()), "ratio": float(fall.float().mean())},
        "timeout": {"count": int(timeout.sum()), "ratio": float(timeout.float().mean())},
        "other": {"count": int(other.sum()), "ratio": float(other.float().mean())},
    }
    for distance in (2, 5, 10):
        reached = displacement >= distance
        metrics[f">={distance}m"] = {"count": int(reached.sum()), "ratio": float(reached.float().mean())}
    out_of_terrain = (terminal_x < terrain_x_bounds[0]) | (terminal_x > terrain_x_bounds[1])
    metrics["out_of_terrain_x"] = {
        "count": int(out_of_terrain.sum()),
        "ratio": float(out_of_terrain.float().mean()),
    }

    decomposition = {name: stats(components[:, index].cpu().numpy()) for index, name in enumerate(terms)}
    decomposition["total"] = stats(returns.cpu().numpy())
    summary = {
        "stage": args.stage,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "evaluation_seed": 24,
        "evaluation_num_envs": 100,
        "observation": "59-D proprio + 48x64 depth image; 64-D CNN embedding; no separate 4-D binary contact",
        "metrics": metrics,
        "reward_components": decomposition,
        "max_step_reward_residual": max_step_error,
        "max_episode_reward_residual": float(residual.max()),
        "mean_episode_reward_residual": float(residual.mean()),
        "terrain_x_bounds": terrain_x_bounds,
        "terrain_mesh_sha256": terrain_mesh_sha256,
        "first_episode_only": True,
        "terminal_step_included": True,
        "post_reset_rewards_excluded": True,
    }

    write_csv(args.output / "episode_metrics.csv", rows)
    write_csv(
        args.output / "main_metrics.csv",
        [{"metric": name, **value} for name, value in metrics.items() if "mean" in value],
    )
    write_csv(
        args.output / "reward_components.csv",
        [{"component": name, **value} for name, value in decomposition.items()],
    )
    (args.output / "evaluation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    env.close()


try:
    main()
finally:
    app.close()
