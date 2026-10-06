"""Final HeightScan + Contact configuration with modified reward.

HeightScan and explicit contact observation are adapted from the existing
IsaacLab_RS Assignment 1 HeightScan+Contact implementation. Training and
inference both reuse the preserved Team1 TotalReward without changing its code.
"""
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.classic.ant.ant_contact_observations import (
    configure_foot_contacts, foot_contact_term,
)
from .ablation_env_cfg import AblationHeightScanCfg, AblationHeightScanObservationsCfg
from .ant_env_cfg import RewardsCfg


@configclass
class Stage2HeightScanContactObservationsCfg(AblationHeightScanObservationsCfg):
    @configclass
    class PolicyCfg(AblationHeightScanObservationsCfg.PolicyCfg):
        foot_contacts = foot_contact_term()

    policy: PolicyCfg = PolicyCfg()


@configclass
class Stage2HeightScanContactCfg(AblationHeightScanCfg):
    observations: Stage2HeightScanContactObservationsCfg = Stage2HeightScanContactObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4096
        configure_foot_contacts(self.scene)
