"""Canonical budget and PPO; fresh contact-feedback run."""
from isaaclab.utils import configclass
from .ablation_budget_ppo_cfg import AblationBudgetPPORunnerCfg


@configclass
class HeightScanContactStockPPORunnerCfg(AblationBudgetPPORunnerCfg):
    run_name = "ablation_heightscan_contact_stock_s42_e4096_n32_i1000"
