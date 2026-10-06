from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, RewardTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

#########################
# Reward weights
#########################

PROGRESS_WEIGHT = 2.5
ALIVE_WEIGHT = 0.5
UPRIGHT_WEIGHT = 0.05
HEADING_WEIGHT = 1.5
CONTACT_WEIGHT = 1.0

ACTION_WEIGHT = -0.005
ENERGY_WEIGHT = -0.15
JOINT_VEL_WEIGHT = -0.001
JOINT_LIMIT_WEIGHT = -0.5
FOOT_SLIP_WEIGHT = -0.07

#########################
# Contact parameter
#########################

CONTACT_FORCE_THRESHOLD = 5.0
# PRINT_CONTACT_FORCES = True
# CONTACT_PRINT_INTERVAL = 1

#########################
# Total reward
#########################


class TotalReward(ManagerTermBase):
    """Compute the complete Ant reward in one manager term."""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        super().__init__(cfg, env)

        robot: Articulation = env.scene["robot"]
        self.target_pos = torch.tensor((1000.0, 0.0, 0.0), device=env.device)
        self.potentials = torch.zeros(env.num_envs, device=env.device)
        self.prev_potentials = torch.zeros_like(self.potentials)
        contact_sensor = env.scene.sensors["contact_forces"]
        self.foot_names = contact_sensor.body_names
        self.foot_body_ids, _ = robot.find_bodies(contact_sensor.body_names, preserve_order=True)
        self.episode_sums = {
            "progress": torch.zeros(env.num_envs, device=env.device),
            "alive": torch.zeros(env.num_envs, device=env.device),
            "upright": torch.zeros(env.num_envs, device=env.device),
            "move_to_target": torch.zeros(env.num_envs, device=env.device),
            "foot_contact": torch.zeros(env.num_envs, device=env.device),
            "action_l2": torch.zeros(env.num_envs, device=env.device),
            "energy": torch.zeros(env.num_envs, device=env.device),
            "joint_pos_limits": torch.zeros(env.num_envs, device=env.device),
            "joint_velocity": torch.zeros(env.num_envs, device=env.device),
            "foot_slip": torch.zeros(env.num_envs, device=env.device),
        }

        gear_ratio = torch.full((env.num_envs, robot.num_joints), 15.0, device=env.device)
        self.gear_ratio_scaled = gear_ratio / torch.max(gear_ratio)

    def reset(self, env_ids: torch.Tensor):
        robot: Articulation = self._env.scene["robot"]
        root_pos = robot.data.root_pos_w[env_ids, :3]
        to_target = self.target_pos - root_pos

        for name, episode_sum in self.episode_sums.items():
            episode_sum_avg = torch.mean(episode_sum[env_ids])
            self._env.extras["log"][f"Episode_Reward/{name}"] = (
                episode_sum_avg / self._env.max_episode_length_s
            )
            episode_sum[env_ids] = 0.0

        self.potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self._env.step_dt
        self.prev_potentials[env_ids] = self.potentials[env_ids]

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        robot: Articulation = env.scene["robot"]
        contact_sensor = env.scene.sensors["contact_forces"]

        root_pos = robot.data.root_pos_w
        root_quat = robot.data.root_quat_w
        projected_gravity = robot.data.projected_gravity_b
        joint_pos = robot.data.joint_pos
        joint_vel = robot.data.joint_vel
        soft_joint_pos_limits = robot.data.soft_joint_pos_limits
        foot_lin_vel = robot.data.body_lin_vel_w[:, self.foot_body_ids]
        contact_force_history = contact_sensor.data.net_forces_w_history
        action = env.action_manager.action
        terminated = env.termination_manager.terminated

        #########################
        # Progress reward
        #########################

        to_target = self.target_pos - root_pos[:, :3]
        to_target[:, 2] = 0.0

        self.prev_potentials[:] = self.potentials
        self.potentials[:] = -torch.norm(to_target, p=2, dim=-1) / env.step_dt
        progress_reward = self.potentials - self.prev_potentials

        #########################
        # Alive reward
        #########################

        alive_reward = (~terminated).float()

        #########################
        # Upright reward
        #########################

        up_projection = -projected_gravity[:, 2]
        upright_reward = (up_projection > 0.93).float()

        #########################
        # Heading reward
        #########################

        target_direction = math_utils.normalize(to_target)
        heading = math_utils.quat_apply(root_quat, robot.data.FORWARD_VEC_B)
        heading_projection = torch.bmm(
            heading.view(env.num_envs, 1, 3),
            target_direction.view(env.num_envs, 3, 1),
        ).view(env.num_envs)
        heading_reward = torch.where(heading_projection > 0.8, 1.0, heading_projection / 0.8)

        #########################
        # Action penalty
        #########################

        action_penalty = torch.sum(torch.square(action), dim=1)

        #########################
        # Energy penalty
        #########################

        energy_penalty = torch.sum(
            torch.abs(action * joint_vel * self.gear_ratio_scaled),
            dim=-1,
        )

        #########################
        # Joint velocity penalty
        #########################

        vel_penalty = torch.sum(torch.square(joint_vel), dim=-1)

        #########################
        # Joint limit penalty
        #########################

        joint_pos_scaled = math_utils.scale_transform(
            joint_pos,
            soft_joint_pos_limits[..., 0],
            soft_joint_pos_limits[..., 1],
        )
        joint_limit_threshold = 0.99

        joint_limit_violation = (torch.abs(joint_pos_scaled) - joint_limit_threshold) / (1 - joint_limit_threshold)
        joint_limit_violation = joint_limit_violation * self.gear_ratio_scaled
        joint_limit_penalty = torch.sum((torch.abs(joint_pos_scaled) > joint_limit_threshold) * joint_limit_violation, dim=-1)

        #########################
        # foot contact reward
        #########################

        vertical_contact_force = contact_force_history[..., 2].amax(dim=1)
        feet_in_contact = vertical_contact_force > CONTACT_FORCE_THRESHOLD

        contact_count = feet_in_contact.sum(dim=1)
        contact_reward = (contact_count >= 2).float()

        # if PRINT_CONTACT_FORCES and env.common_step_counter % CONTACT_PRINT_INTERVAL == 0:
        #     force_values = vertical_contact_force[0].detach().cpu().tolist()
        #     contact_values = feet_in_contact[0].detach().cpu().tolist()
        #     foot_states = ", ".join(
        #         f"{name}={force:.2f}N ({'contact' if in_contact else 'air'})"
        #         for name, force, in_contact in zip(self.foot_names, force_values, contact_values)
        #     )
        #     print(f"[CONTACT][step={env.common_step_counter}] {foot_states}", flush=True)

        #########################
        # Foot slip penalty 
        #########################

        foot_speed_xy = torch.norm(foot_lin_vel[..., :2], dim=-1)
        foot_slip_penalty = torch.sum(foot_speed_xy * feet_in_contact.float(), dim=1)

        #########################
        # Total reward
        #########################

        weighted_rewards = {
            "progress": PROGRESS_WEIGHT * progress_reward,
            "alive": ALIVE_WEIGHT * alive_reward,
            "upright": UPRIGHT_WEIGHT * upright_reward,
            "move_to_target": HEADING_WEIGHT * heading_reward,
            "foot_contact": CONTACT_WEIGHT * contact_reward,
            "action_l2": ACTION_WEIGHT * action_penalty,
            "energy": ENERGY_WEIGHT * energy_penalty,
            "joint_velocity": JOINT_VEL_WEIGHT * vel_penalty,
            "joint_pos_limits": JOINT_LIMIT_WEIGHT * joint_limit_penalty,
            "foot_slip": FOOT_SLIP_WEIGHT * foot_slip_penalty,
        }

        for name, reward in weighted_rewards.items():
            self.episode_sums[name] += reward * env.step_dt

        total_reward = sum(weighted_rewards.values())
        return total_reward
