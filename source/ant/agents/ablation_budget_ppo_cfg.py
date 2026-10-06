"""Canonical 4096 x 32 x 1000 budget; historical runner stays unchanged."""
from isaaclab.utils import configclass
from .ablation_ppo_cfg import AblationHeightScanPPORunnerCfg


@configclass
class AblationBudgetPPORunnerCfg(AblationHeightScanPPORunnerCfg):
    seed = 42
    num_steps_per_env = 32
    max_iterations = 1000
    run_name = "ablation_heightscan_stock_s42_e4096_n32_i1000"
    resume = False
    load_run = None
    load_checkpoint = None


@configclass
class AblationBaselinePPORunnerCfg(AblationBudgetPPORunnerCfg):
    """Stock-observation baseline with a distinct output run name."""

    run_name = "ablation_baseline_stock_s42_e4096_n32_i1000"
