# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment (similar to OpenAI Gym Ant-v2).
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Ant-rl-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_env_cfg:AntEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntPPORunnerCfg",
    },
)

gym.register(
    id="Ant-rl-Ablation-HeightScan-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ablation_env_cfg:AblationHeightScanCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ablation_ppo_cfg:AblationHeightScanPPORunnerCfg",
    },
)

gym.register(
    id="Ant-rl-Ablation-HeightScan-Contact-ModifiedReward-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stage2_env_cfg:Stage2HeightScanContactCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.stage2_ppo_cfg:Stage2HeightScanContactPPORunnerCfg",
    },
)

gym.register(
    id="Ant-rl-Ablation-HeightScan-Contact-Stock-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.contact_stock_env_cfg:HeightScanContactStockCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.contact_stock_ppo_cfg:HeightScanContactStockPPORunnerCfg",
    },
)

# Stock observation and reward foundation for the first ablation stage.
gym.register(
    id="Ant-rl-Ablation-Baseline-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ablation_env_cfg:AblationBaseCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ablation_budget_ppo_cfg:AblationBaselinePPORunnerCfg",
    },
)
