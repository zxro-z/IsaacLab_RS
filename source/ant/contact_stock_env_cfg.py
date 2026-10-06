"""Contact-feedback comparison: canonical HeightScan and stock reward.

Explicit contact semantics reuse the validated IsaacLab_RS HeightScan+Contact
implementation; contact is added only to observations, never to rewards.
"""
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.classic.ant.ant_contact_observations import configure_foot_contacts
from .ablation_env_cfg import AblationHeightScanCfg
from .stage2_env_cfg import Stage2HeightScanContactObservationsCfg


@configclass
class HeightScanContactStockCfg(AblationHeightScanCfg):
    observations: Stage2HeightScanContactObservationsCfg = Stage2HeightScanContactObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4096
        configure_foot_contacts(self.scene)
