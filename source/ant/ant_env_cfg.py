# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.classic.humanoid.mdp as mdp

# import isaaclab.terrains as terrain_gen
import torch

from . import depth_obs, rewards

##
# Pre-defined configs
##
from isaaclab_assets.robots.ant import ANT_CFG  # isort: skip

# TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
#     seed=42,
#     curriculum=False,
#     size=(8.0, 8.0),
#     border_width=20.0,
#     num_rows=25,
#     num_cols=10,
#     use_cache=False,
#     sub_terrains={
#         "random_grid": terrain_gen.MeshRandomGridTerrainCfg(
#             proportion=1.0,
#             grid_width=0.45,
#             grid_height_range=(0.05, 0.20),
#             platform_width=2.0,
#             holes=False,
#         ),
#     },
# )


#########################
# random Terrain settings
#########################

TERRAIN_CFG = ROUGH_TERRAINS_CFG.copy()

TERRAIN_CFG.size = (10.0, 10.0)
TERRAIN_CFG.num_rows = 20
TERRAIN_CFG.num_cols = 10
TERRAIN_CFG.seed = 42
TERRAIN_CFG.curriculum = False
TERRAIN_CFG.border_width = 2.0

# 생성 확률
TERRAIN_CFG.sub_terrains["pyramid_stairs"].proportion = 0.2
TERRAIN_CFG.sub_terrains["hf_pyramid_slope"].proportion = 0.2
TERRAIN_CFG.sub_terrains["boxes"].proportion = 0.2
TERRAIN_CFG.sub_terrains["pyramid_stairs_inv"].proportion = 0.2
TERRAIN_CFG.sub_terrains["hf_pyramid_slope_inv"].proportion = 0.2
TERRAIN_CFG.sub_terrains["random_rough"].proportion = 0.0

# 높이
TERRAIN_CFG.sub_terrains["pyramid_stairs"].step_height_range = (0.03, 0.07)
TERRAIN_CFG.sub_terrains["pyramid_stairs_inv"].step_height_range = (0.03, 0.07)
TERRAIN_CFG.sub_terrains["boxes"].grid_height_range = (0.02, 0.10)

# 기울기
TERRAIN_CFG.sub_terrains["hf_pyramid_slope"].slope_range = (0.0, 0.20)
TERRAIN_CFG.sub_terrains["hf_pyramid_slope_inv"].slope_range = (0.0, 0.20)

# border_width 설정
TERRAIN_CFG.sub_terrains["pyramid_stairs"].border_width = 0
TERRAIN_CFG.sub_terrains["pyramid_stairs_inv"].border_width = 0
TERRAIN_CFG.sub_terrains["hf_pyramid_slope"].border_width = 0
TERRAIN_CFG.sub_terrains["hf_pyramid_slope_inv"].border_width = 0

# block 중앙의 flat 영역
TERRAIN_CFG.sub_terrains["pyramid_stairs"].platform_width = 1.0
TERRAIN_CFG.sub_terrains["pyramid_stairs_inv"].platform_width = 1.0
TERRAIN_CFG.sub_terrains["boxes"].platform_width = 1.0
TERRAIN_CFG.sub_terrains["hf_pyramid_slope"].platform_width = 1.0
TERRAIN_CFG.sub_terrains["hf_pyramid_slope_inv"].platform_width = 1.0
#########################
#
#########################

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with an ant robot."""

    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TERRAIN_CFG,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,        # 바닥의 마찰계수는 1로 설정, 로봇의 마찰계수만 변경 - 환경마다 다른 마찰계수
            dynamic_friction=1.0,
            restitution=0.0,
        ), 
        debug_vis=False,
    )

    # robot
    robot = ANT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.spawn.usd_path = robot.spawn.usd_path.replace("ant_instanceable.usd", "ant.usd")
    robot.spawn.activate_contact_sensors = True
    robot.spawn.copy_from_source = True

    # depth camera  64x48 / 15Hz / 0.1~5.0m
    depth_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso/DepthCamera",
        update_period=1.0 / 15.0,
        height=48,
        width=64,
        data_types=["distance_to_camera"],
        depth_clipping_behavior="max",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=16.0,
            focus_distance=5.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 5.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.25, 0.0, 0.12),
            rot=(0.9962, 0.0, 0.0872, 0.0),
            convention="world",
        ),
    )

    # foot contacts
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_foot",
        history_length=3,
        track_air_time=False,
        force_threshold=1.0,
        debug_vis=True,
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

 
##
# MDP settings
##

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=[".*"], scale=7.5)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""

        # base_height = ObsTerm(func=mdp.base_pos_z)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        base_yaw_roll = ObsTerm(func=mdp.base_yaw_roll)
        base_angle_to_target = ObsTerm(func=mdp.base_angle_to_target, params={"target_pos": (1000.0, 0.0, 0.0)})
        base_up_proj = ObsTerm(func=mdp.base_up_proj)
        base_heading_proj = ObsTerm(func=mdp.base_heading_proj, params={"target_pos": (1000.0, 0.0, 0.0)})
        joint_pos_norm = ObsTerm(func=mdp.joint_pos_limit_normalized)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.2)
        feet_body_forces = ObsTerm(
            func=mdp.body_incoming_wrench,
            scale=0.1,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=["front_left_foot", "front_right_foot", "left_back_foot", "right_back_foot"]
                )
            },
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class DepthCfg(ObsGroup):
        """Depth images for the CNN encoder."""

        image = ObsTerm(
            func=depth_obs.normalized_depth,
            params={
                "sensor_cfg": SceneEntityCfg("depth_camera"),
                "near_distance": 0.1,
                "far_distance": 5.0,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    depth: DepthCfg = DepthCfg()


def randomize_robot_friction(env, env_ids, asset_cfg, min_fric, max_fric):
    robot = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    materials = robot.root_physx_view.get_material_properties()

    mu_static = torch.empty((len(env_ids), 1), device="cpu").uniform_(min_fric, max_fric)
    mu_dynamic = 0.8 * mu_static  # dynamic friction is 80% of static friction

    materials[env_ids, :, 0] = mu_static   # static friction
    materials[env_ids, :, 1] = mu_dynamic   # dynamic friction
    materials[env_ids, :, 2] = 0.0

    robot.root_physx_view.set_material_properties(materials, env_ids)


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}},
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-0.1, 0.1),
        },
    )
    # 로봇의 마찰계수를 랜덤으로 적용해서 학습   0.3 ~ 1.0
    random_friction = EventTerm(
        func=randomize_robot_friction,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "min_fric": 0.3,
            "max_fric": 1.0 
        }
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    total_reward = RewTerm(func=rewards.TotalReward, weight=1.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Terminate if the episode length is exceeded
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Terminate if the robot falls
    # torso_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.31})
    # (3) Terminate if the body z-axis points downward
    body_z_down = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.pi / 2})


@configclass
class AntEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the MuJoCo-style Ant walking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=256, env_spacing=5.0, clone_in_fabric=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 16.0
        # simulation settings
        self.sim.dt = 1 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        # default friction material
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.restitution = 0.0
