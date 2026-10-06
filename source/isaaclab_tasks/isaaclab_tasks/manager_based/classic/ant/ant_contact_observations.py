# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Ant foot-contact observations and optional diagnostics. No reward functions."""

import torch

from isaaclab.managers import ObservationTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg

FOOT_BODY_NAMES = ["front_left_foot", "front_right_foot", "left_back_foot", "right_back_foot"]
CONTACT_FORCE_THRESHOLD = 1.0  # N; matches the native ContactSensor's default meaningful-contact threshold.


def foot_contact_state(env, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """Return (N, 4) float contact states, in the manager-resolved foot order."""
    forces = env.scene[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids, :]
    return (torch.linalg.vector_norm(forces, dim=-1) > threshold).to(forces.dtype)


def foot_contact_term() -> ObservationTermCfg:
    return ObservationTermCfg(
        func=foot_contact_state,
        params={
            "sensor_cfg": SceneEntityCfg("feet_contacts", body_names=FOOT_BODY_NAMES.copy(), preserve_order=True),
            "threshold": CONTACT_FORCE_THRESHOLD,
        },
    )


def foot_contact_sensor_cfg() -> ContactSensorCfg:
    return ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/(" + "|".join(FOOT_BODY_NAMES) + ")",
        update_period=0.0,
        history_length=0,
        track_air_time=False,
        debug_vis=False,
    )


def configure_foot_contacts(scene_cfg):
    """Enable native reporting on this config copy only; reuse its sensor if present."""
    scene_cfg.robot = scene_cfg.robot.copy()
    scene_cfg.robot.spawn.activate_contact_sensors = True
    if getattr(scene_cfg, "feet_contacts", None) is None:
        scene_cfg.feet_contacts = foot_contact_sensor_cfg()
    # SensorBase resolves parent prims through USD, like the existing ray casters.
    scene_cfg.clone_in_fabric = False


def add_foot_contacts_for_evaluation(env_cfg):
    """Opt-in 64-D/127-D runtime config; no observation padding or reward changes."""
    from .ant_terrain_heightscan_env_cfg import AntFootContactObservationsCfg, AntHeightScanContactObservationsCfg

    configure_foot_contacts(env_cfg.scene)
    if getattr(env_cfg.observations.policy, "height_scan", None) is not None:
        env_cfg.observations = AntHeightScanContactObservationsCfg()
    else:
        env_cfg.observations = AntFootContactObservationsCfg()


def log_foot_contacts(env):
    """One-shot debug output, outside the observation path (printing synchronizes)."""
    sensor_cfg = SceneEntityCfg("feet_contacts", body_names=FOOT_BODY_NAMES.copy(), preserve_order=True)
    sensor_cfg.resolve(env.scene)
    forces = env.scene[sensor_cfg.name].data.net_forces_w[0, sensor_cfg.body_ids, :]
    state = foot_contact_state(env, sensor_cfg, CONTACT_FORCE_THRESHOLD)[0]
    print(f"[CONTACT DEBUG] ordered_foot_names: {FOOT_BODY_NAMES}")
    print(f"[CONTACT DEBUG] force_magnitudes_N: {torch.linalg.vector_norm(forces, dim=-1).tolist()}")
    print(f"[CONTACT DEBUG] foot_contacts: {state.tolist()}")
    shape = (env.num_envs, *env.observation_manager.group_obs_dim["policy"])
    print(f"[CONTACT DEBUG] policy_observation_shape: {shape}")
