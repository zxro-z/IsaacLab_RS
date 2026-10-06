"""Keep Team1 PPO and post-embedding MLP; use a plain concatenated MLP."""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg
from .rsl_rl_ppo_cfg import AntPPORunnerCfg


@configclass
class AblationHeightScanPPORunnerCfg(AntPPORunnerCfg):
    seed = 42
    max_iterations = 10000
    experiment_name = "observation_ablation"
    run_name = "ablation_heightscan_stock_s42"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[400, 200, 100],
        critic_hidden_dims=[400, 200, 100],
        activation="elu",
    )
