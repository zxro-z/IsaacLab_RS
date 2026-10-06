"""Team1 dynamics with canonical IsaacLab_RS stock rewards and HeightScan.

HeightScan integration is adapted from the IsaacLab_RS Assignment 1 HeightScan
implementation. Historical Ant-rl-v0 and its custom reward are unchanged.
"""

from isaaclab.managers import ObservationTermCfg as ObsTerm, SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import RewardsCfg as StockAntRewardsCfg

from .ant_env_cfg import AntEnvCfg, ObservationsCfg, mdp


@configclass
class AblationBaseObservationsCfg(ObservationsCfg):
    depth = None


@configclass
class AblationHeightScanObservationsCfg(AblationBaseObservationsCfg):
    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5},
            clip=(-1.0, 1.0),
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class AblationBaseCfg(AntEnvCfg):
    """Shared dynamics/reward foundation for future observation variants.

    Evaluation uses this same Team1 environment, seed 24 and 100 environments.
    Sensor-specific changes belong in subclasses; frozen settings are inherited.
    """

    rewards: StockAntRewardsCfg = StockAntRewardsCfg()
    observations: AblationBaseObservationsCfg = AblationBaseObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.seed = 42
        self.scene.num_envs = 2048
        self.scene.depth_camera = None


@configclass
class AblationHeightScanCfg(AblationBaseCfg):
    observations: AblationHeightScanObservationsCfg = AblationHeightScanObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso",
            offset=RayCasterCfg.OffsetCfg(pos=(0.8, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=(1.6, 1.2)),
            mesh_prim_paths=["/World/ground"],
            debug_vis=False,
        )
