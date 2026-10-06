from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg


#########################
# Depth observation
#########################


def normalized_depth(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("depth_camera"),
    near_distance: float = 0.1,
    far_distance: float = 5.0,
) -> torch.Tensor:
    """Return clipped depth images normalized to the range [0, 1]."""
    camera = env.scene.sensors[sensor_cfg.name]
    depth = camera.data.output["distance_to_camera"].clone()
    depth = torch.nan_to_num(depth, nan=far_distance, posinf=far_distance, neginf=near_distance)
    depth = depth.clamp_(near_distance, far_distance)
    return (depth - near_distance) / (far_distance - near_distance)
