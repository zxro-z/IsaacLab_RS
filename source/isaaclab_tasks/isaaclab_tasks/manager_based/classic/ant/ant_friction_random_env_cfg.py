# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ant environment configuration with startup-time robot friction randomization."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .ant_env_cfg import AntEnvCfg, EventCfg, mdp


@configclass
class AntFrictionRandomEventCfg(EventCfg):
    """Ant events with randomized rigid-body material properties."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )


@configclass
class AntFrictionRandomEnvCfg(AntEnvCfg):
    """Baseline Ant environment with robot friction randomization only."""

    events: AntFrictionRandomEventCfg = AntFrictionRandomEventCfg()
