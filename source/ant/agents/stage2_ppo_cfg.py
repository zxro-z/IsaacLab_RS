"""Shared Stage 2 PPO: only terrain representation varies between arms."""
from isaaclab.utils import configclass
from .ablation_budget_ppo_cfg import AblationBudgetPPORunnerCfg


@configclass
class Stage2HeightScanContactPPORunnerCfg(AblationBudgetPPORunnerCfg):
    run_name = 'ablation_heightscan_contact_modified_s42_e4096_n32_i1000'
