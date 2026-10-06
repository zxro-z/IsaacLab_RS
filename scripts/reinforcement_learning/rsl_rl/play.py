# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import csv
import sys
from collections import deque

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
import ant  # noqa: F401

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playback.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--record_vel",
    action="store_true",
    default=False,
    help="Print and save each environment's mean velocity when an episode ends.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# depth observations always require camera rendering
args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.classic.humanoid.mdp as mdp
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from ant.depth_actor_critic import DepthActorCritic, register_depth_actor_critic

register_depth_actor_critic()


@configclass
class EvalRewardsCfg:
    # 기존 Ant reward

    progress = RewTerm(
        func=mdp.progress_reward,
        weight=1.0,
        params={"target_pos": (1000.0, 0.0, 0.0)},
    )

    alive = RewTerm(
        func=mdp.is_alive,
        weight=0.5,
    )

    upright = RewTerm(
        func=mdp.upright_posture_bonus,
        weight=0.1,
        params={"threshold": 0.93},
    )

    move_to_target = RewTerm(
        func=mdp.move_to_target_bonus,
        weight=0.5,
        params={
            "threshold": 0.8,
            "target_pos": (1000.0, 0.0, 0.0),
        },
    )

    action_l2 = RewTerm(
        func=mdp.action_l2,
        weight=-0.005,
    )

    energy = RewTerm(
        func=mdp.power_consumption,
        weight=-0.05,
        params={"gear_ratio": {".*": 15.0}},
    )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    env_cfg.rewards = EvalRewardsCfg()

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during playback.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    policy_nn = runner.alg.policy

    if isinstance(policy_nn, DepthActorCritic):
        print("[INFO] Skipping JIT/ONNX export for the multimodal depth policy.")
    else:
        normalizer = getattr(policy_nn, "actor_obs_normalizer", None)
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    num_envs = env.unwrapped.num_envs
    device = env.unwrapped.device

    episode_returns = torch.zeros(num_envs, device=device)
    pending_returns = [deque() for _ in range(num_envs)]

    eval_episode = 1
    timestep = 0

    velocity_log_file = None
    if args_cli.record_vel:
        velocity_log_path = os.path.join(log_dir, "play_episode_velocities.csv")
        write_header = not os.path.exists(velocity_log_path) or os.path.getsize(velocity_log_path) == 0
        velocity_log_file = open(velocity_log_path, "a", newline="", encoding="utf-8")
        velocity_log_writer = csv.writer(velocity_log_file)
        if write_header:
            velocity_log_writer.writerow(
                [
                    "session",
                    "env_id",
                    "episode",
                    "duration_s",
                    "mean_forward_velocity_m_s",
                    "mean_planar_speed_m_s",
                ]
            )

        session = time.strftime("%Y%m%d_%H%M%S")
        forward_velocity_sum = torch.zeros(num_envs, device=device)
        planar_speed_sum = torch.zeros(num_envs, device=device)
        episode_step_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        episode_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        print(f"[INFO] Episode velocity log: {velocity_log_path}")

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            if args_cli.record_vel:
                root_linear_velocity = env.unwrapped.scene["robot"].data.root_lin_vel_w
                forward_velocity = root_linear_velocity[:, 0].clone()
                planar_speed = torch.linalg.vector_norm(root_linear_velocity[:, :2], dim=1).clone()

            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, rewards, dones, _ = env.step(actions)

            episode_returns += rewards

            done_ids = dones.nonzero(as_tuple=False).flatten()

            if len(done_ids) > 0:
                done_env_ids = done_ids.detach().cpu().tolist()
                done_returns = episode_returns[done_ids].detach().cpu().tolist()
                for env_id, episode_return in zip(done_env_ids, done_returns):
                    pending_returns[env_id].append(episode_return)

                episode_returns[done_ids] = 0.0

                # 각 env에서 완료된 episode를 하나씩 모아 동일한 평가 묶음으로 출력
                while all(pending_returns):
                    returns_cpu = torch.tensor([returns.popleft() for returns in pending_returns])

                    print(f"\n========== Evaluation Episode {eval_episode} ==========")

                    for env_id, total_reward in enumerate(returns_cpu.tolist()):
                        print(f"env {env_id:4d}: {total_reward:.3f}")

                    print(
                        f"Mean reward ({num_envs} envs): "
                        f"{returns_cpu.mean().item():.3f}"
                    )

                    print("================================================\n")

                    eval_episode += 1

            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)

            if args_cli.record_vel:
                forward_velocity_sum += forward_velocity
                planar_speed_sum += planar_speed
                episode_step_count += 1

                finished_env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                for env_id in finished_env_ids.tolist():
                    episode_count[env_id] += 1
                    steps = episode_step_count[env_id].item()
                    duration = steps * dt
                    mean_forward_velocity = (forward_velocity_sum[env_id] / steps).item()
                    mean_planar_speed = (planar_speed_sum[env_id] / steps).item()

                    print(
                        f"[VELOCITY] env={env_id} episode={episode_count[env_id].item()} "
                        f"duration={duration:.2f}s forward={mean_forward_velocity:.3f}m/s "
                        f"planar_speed={mean_planar_speed:.3f}m/s"
                    )
                    velocity_log_writer.writerow(
                        [
                            session,
                            env_id,
                            episode_count[env_id].item(),
                            duration,
                            mean_forward_velocity,
                            mean_planar_speed,
                        ]
                    )
                    velocity_log_file.flush()

                forward_velocity_sum[finished_env_ids] = 0.0
                planar_speed_sum[finished_env_ids] = 0.0
                episode_step_count[finished_env_ids] = 0
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if velocity_log_file is not None:
        velocity_log_file.close()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
