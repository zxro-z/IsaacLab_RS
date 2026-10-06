# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluation-only terrain OOD configurations for the Ant task.

These environments are for evaluation only, not training or model selection.
"""

import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainGenerator, TerrainGeneratorCfg, TerrainImporter, TerrainImporterCfg
from isaaclab.terrains.height_field.hf_terrains import pyramid_stairs_terrain
from isaaclab.utils import configclass

from .ant_env_cfg import AntEnvCfg, MySceneCfg, TerminationsCfg


def torso_clearance_below_minimum(env, minimum_height: float) -> torch.Tensor:
    """Evaluation-only fall check using the terrain surface directly below the torso.

    Keep the original 0.31 m vertical clearance semantics on a flat plane. A missing
    terrain hit ends the episode rather than silently using the spawn platform height.
    """
    terrain_z = env.scene["torso_ground_ray"].data.ray_hits_w[:, 0, 2]
    clearance = env.scene["robot"].data.root_pos_w[:, 2] - terrain_z
    return ~torch.isfinite(terrain_z) | (clearance < minimum_height)


@configclass
class AntOODTerrainTerminationsCfg(TerminationsCfg):
    """Evaluation-only local terrain clearance; preserve time_out and the term name."""

    torso_height = DoneTerm(func=torso_clearance_below_minimum, params={"minimum_height": 0.31})


def uphill_stairs_with_center_origin(difficulty: float, cfg):
    """Keep native stair geometry but reference the actual central spawn surface.

    The native origin is the maximum of a central window; on inverted stairs that
    window includes the adjacent higher step rather than only the spawn platform.
    """
    meshes, origin = pyramid_stairs_terrain(difficulty, cfg)
    vertices = meshes[0].vertices
    center_vertex = np.argmin(np.sum((vertices[:, :2] - origin[:2]) ** 2, axis=1))
    origin[2] = vertices[center_vertex, 2]
    return meshes, origin


class AntOODTerrainGenerator(TerrainGenerator):
    """Translate the entire evaluation terrain so its spawn platforms lie at world Z=0."""

    def __init__(self, cfg: TerrainGeneratorCfg, device: str = "cpu"):
        super().__init__(cfg, device)
        platform_z = float(self.terrain_origins[0, 0, 2])
        if not np.allclose(self.terrain_origins[..., 2], platform_z):
            raise ValueError("Ant OOD terrain requires a common platform height for a global translation.")
        offset = np.array([0.0, 0.0, -platform_z])
        # Translate geometry and origins before TerrainImporter creates collision meshes and env origins.
        self.terrain_mesh.apply_translation(offset)
        self.terrain_origins += offset
        for patches in self.flat_patches.values():
            patches[..., 2] -= platform_z


class AntOODTerrainImporter(TerrainImporter):
    """Use a central patch for single-env inspection; retain native multi-env assignment."""

    def configure_env_origins(self, origins: np.ndarray | torch.Tensor | None = None):
        super().configure_env_origins(origins)
        if self.cfg.num_envs == 1 and self.terrain_origins is not None:
            num_rows, num_cols = self.terrain_origins.shape[:2]
            self.terrain_levels[0] = (num_rows - 1) // 2
            self.terrain_types[0] = (num_cols - 1) // 2
            self.env_origins[0] = self.terrain_origins[self.terrain_levels[0], self.terrain_types[0]]


def _terrain_cfg(sub_terrain: terrain_gen.SubTerrainBaseCfg) -> TerrainImporterCfg:
    """Create 100 deterministic evaluation terrain origins."""
    return TerrainImporterCfg(
        class_type=AntOODTerrainImporter,
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            class_type=AntOODTerrainGenerator,
            size=(8.0, 8.0),
            num_rows=10,
            num_cols=10,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            difficulty_range=(1.0, 1.0),
            use_cache=False,
            sub_terrains={"ood": sub_terrain},
        ),
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )


@configclass
class AntOODTerrainSceneCfg(MySceneCfg):
    """Shared evaluation-only torso ray for terrain-relative fall detection."""

    # Use USD cloning in these evaluation configs: SensorBase counts parent prims
    # through USD, while Fabric-only clones are visible only to the physics view.
    # A single vertical ray follows the live torso XY, including when the torso tilts.
    # This sensor is used only for termination and is not added to policy observations.
    torso_ground_ray = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(0.0, 0.0)),
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )


@configclass
class AntOODRampUpSceneCfg(AntOODTerrainSceneCfg):
    """Ant scene with an inverted pyramid slope, rising from its center."""

    terrain = _terrain_cfg(
        terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=1.0, slope_range=(0.15, 0.15), platform_width=2.0, border_width=0.25
        )
    )


@configclass
class AntOODRampDownSceneCfg(AntOODTerrainSceneCfg):
    """Ant scene with a pyramid slope, descending from its center."""

    terrain = _terrain_cfg(
        terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=1.0, slope_range=(0.15, 0.15), platform_width=2.0, border_width=0.25
        )
    )


@configclass
class AntOODStairsSceneCfg(AntOODTerrainSceneCfg):
    """Ant scene with inverted pyramid stairs, ascending outward in +X."""

    terrain = _terrain_cfg(
        terrain_gen.HfInvertedPyramidStairsTerrainCfg(
            function=uphill_stairs_with_center_origin,
            proportion=1.0,
            step_height_range=(0.10, 0.10),
            step_width=0.40,
            platform_width=2.0,
        )
    )


@configclass
class AntOODTerrainEnvCfg(AntEnvCfg):
    """Shared evaluation-only fall semantics for every terrain OOD task."""

    terminations: AntOODTerrainTerminationsCfg = AntOODTerrainTerminationsCfg()


@configclass
class AntOODRampUpEnvCfg(AntOODTerrainEnvCfg):
    """Evaluation-only Ant environment on a rising ramp terrain."""

    scene: InteractiveSceneCfg = AntOODRampUpSceneCfg(num_envs=4096, env_spacing=5.0, clone_in_fabric=False)


@configclass
class AntOODRampDownEnvCfg(AntOODTerrainEnvCfg):
    """Evaluation-only Ant environment on a descending ramp terrain."""

    scene: InteractiveSceneCfg = AntOODRampDownSceneCfg(num_envs=4096, env_spacing=5.0, clone_in_fabric=False)


@configclass
class AntOODStairsEnvCfg(AntOODTerrainEnvCfg):
    """Evaluation-only Ant environment on stairs terrain."""

    scene: InteractiveSceneCfg = AntOODStairsSceneCfg(num_envs=4096, env_spacing=5.0, clone_in_fabric=False)
