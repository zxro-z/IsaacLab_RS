# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Training-only Ant terrain/perception ablation; existing Ant tasks are unchanged."""

import numpy as np

import isaaclab.terrains as terrain_gen
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainGenerator, TerrainGeneratorCfg
from isaaclab.utils import configclass

from .ant_contact_observations import configure_foot_contacts, foot_contact_sensor_cfg, foot_contact_term
from .ant_env_cfg import MySceneCfg, ObservationsCfg, mdp
from .ant_friction_random_env_cfg import AntFrictionRandomEnvCfg, AntFrictionRandomEventCfg
from .ant_ood_terrain_env_cfg import AntOODTerrainSceneCfg, AntOODTerrainTerminationsCfg


class AntTrainingTerrainGenerator(TerrainGenerator):
    """Normalize each native patch's actual central surface before tiling.

    Different slopes/stairs have different native platform elevations. Translating
    mesh and origin together preserves geometry, while retaining the Ant's original
    reset pose and world-Z observation convention at every spawn platform.
    """

    def _get_terrain_mesh(self, difficulty, cfg):
        mesh, origin = super()._get_terrain_mesh(difficulty, cfg)
        nearest = np.argmin(np.sum((mesh.vertices[:, :2] - origin[:2]) ** 2, axis=1))
        platform_z = float(mesh.vertices[nearest, 2])
        mesh.apply_translation((0.0, 0.0, -platform_z))
        origin[2] = 0.0
        return mesh, origin


def height_scanner_cfg() -> RayCasterCfg:
    """63 yaw-aligned rays: X=0..1.6 m ahead, Y=-0.6..0.6 m, 0.2 m spacing."""
    return RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso",
        offset=RayCasterCfg.OffsetCfg(pos=(0.8, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=(1.6, 1.2)),
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )


def _training_terrain_cfg():
    # Keep the baseline plane's importer/material settings, replacing only geometry.
    terrain = MySceneCfg().terrain.replace(terrain_type="generator")
    terrain.terrain_generator = TerrainGeneratorCfg(
        class_type=AntTrainingTerrainGenerator,
        size=(8.0, 8.0),
        num_rows=10,
        num_cols=10,
        border_width=20.0,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        curriculum=False,
        difficulty_range=(0.0, 1.0),
        use_cache=False,
        sub_terrains={
            "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.30),
            "up_easy": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                proportion=0.15, slope_range=(0.0, 0.05), platform_width=2.0, border_width=0.25
            ),
            "down_easy": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=0.15, slope_range=(0.0, 0.05), platform_width=2.0, border_width=0.25
            ),
            "up_medium": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                proportion=0.10, slope_range=(0.05, 0.10), platform_width=2.0, border_width=0.25
            ),
            "down_medium": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=0.10, slope_range=(0.05, 0.10), platform_width=2.0, border_width=0.25
            ),
            "up_harder": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                proportion=0.05, slope_range=(0.10, 0.15), platform_width=2.0, border_width=0.25
            ),
            "down_harder": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=0.05, slope_range=(0.10, 0.15), platform_width=2.0, border_width=0.25
            ),
            "low_up_stairs": terrain_gen.HfInvertedPyramidStairsTerrainCfg(
                proportion=0.10, step_height_range=(0.025, 0.05), step_width=0.40, platform_width=2.0
            ),
        },
    )
    return terrain


@configclass
class AntHeightScanObservationsCfg(ObservationsCfg):
    """Append a native torso-relative height scan; preserve all original 60 inputs."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5},
            clip=(-1.0, 1.0),
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class AntFootContactObservationsCfg(ObservationsCfg):
    """Original 60-D Ant inputs plus four ordered contact states."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        foot_contacts = foot_contact_term()

    policy: PolicyCfg = PolicyCfg()


@configclass
class AntHeightScanContactObservationsCfg(AntHeightScanObservationsCfg):
    """Keep all 123 existing inputs and append only four contact states."""

    @configclass
    class PolicyCfg(AntHeightScanObservationsCfg.PolicyCfg):
        foot_contacts = foot_contact_term()

    policy: PolicyCfg = PolicyCfg()


def add_height_scan_for_evaluation(env_cfg):
    """Add the 63-D scan, retaining contact inputs if explicitly configured.

    Only the supplied runtime config instance changes. Terrain, rewards, resets,
    terminations and registered task definitions remain untouched.
    """
    env_cfg.scene.height_scanner = height_scanner_cfg()
    env_cfg.scene.clone_in_fabric = False
    # Preserve an explicitly enabled contact term when composing runtime adapters.
    if getattr(env_cfg.observations.policy, "foot_contacts", None) is not None:
        env_cfg.observations = AntHeightScanContactObservationsCfg()
    else:
        env_cfg.observations = AntHeightScanObservationsCfg()


@configclass
class AntTerrainHeightScanSceneCfg(MySceneCfg):
    terrain = _training_terrain_cfg()
    height_scanner = height_scanner_cfg()
    # A single additional ray gives robust clearance directly below the live torso,
    # independent of the look-ahead grid ordering. Reuse the validated OOD pattern.
    torso_ground_ray = AntOODTerrainSceneCfg().torso_ground_ray


@configclass
class AntTerrainHeightScanContactSceneCfg(AntTerrainHeightScanSceneCfg):
    """Add only foot sensing to the original terrain/height-scan scene."""

    feet_contacts = foot_contact_sensor_cfg()


@configclass
class AntTerrainHeightScanEventsCfg(AntFrictionRandomEventCfg):
    def __post_init__(self):
        # Supported native option, applied only to this new task's independent config.
        self.physics_material.params["make_consistent"] = True


@configclass
class AntTerrainRandomHeightScanEnvCfg(AntFrictionRandomEnvCfg):
    """New training task: randomized geometry/materials, perception, local fall test.

    USD cloning is required: SensorBase counts USD parents, not Fabric-only clones.
    The original PPO entry point infers the larger observation size automatically.
    """

    scene: InteractiveSceneCfg = AntTerrainHeightScanSceneCfg(
        num_envs=4096, env_spacing=5.0, clone_in_fabric=False
    )
    observations: AntHeightScanObservationsCfg = AntHeightScanObservationsCfg()
    events: AntTerrainHeightScanEventsCfg = AntTerrainHeightScanEventsCfg()
    terminations: AntOODTerrainTerminationsCfg = AntOODTerrainTerminationsCfg()


@configclass
class AntTerrainRandomHeightScanContactEnvCfg(AntTerrainRandomHeightScanEnvCfg):
    """Fresh 127-D ablation: original 123 inputs plus four foot contacts.

    Rewards, terrain, resets, termination and simulation settings are inherited
    unchanged. Contact reporting is enabled only on this variant's robot copy.
    """

    scene: InteractiveSceneCfg = AntTerrainHeightScanContactSceneCfg(
        num_envs=4096, env_spacing=5.0, clone_in_fabric=False
    )
    observations: AntHeightScanContactObservationsCfg = AntHeightScanContactObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        configure_foot_contacts(self.scene)
