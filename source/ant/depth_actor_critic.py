from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import MLP, EmpiricalNormalization


#########################
# Depth policy
#########################


class DepthActorCritic(nn.Module):
    """Actor-critic with a shared depth CNN and proprioceptive observations."""

    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=(400, 200, 100),
        critic_hidden_dims=(400, 200, 100),
        activation="elu",
        init_noise_std=1.0,
        noise_std_type="scalar",
        depth_embedding_dim=64,
        proprioception_group="policy",
        depth_group="depth",
        **kwargs,
    ):
        if kwargs:
            print(f"DepthActorCritic ignored arguments: {list(kwargs.keys())}")
        super().__init__()

        self.obs_groups = obs_groups
        self.proprioception_group = proprioception_group
        self.depth_group = depth_group

        proprioception = obs[self.proprioception_group]
        depth = obs[self.depth_group]
        if proprioception.ndim != 2:
            raise ValueError(f"Expected 1D proprioception, received shape {tuple(proprioception.shape)}")
        if depth.ndim != 4 or depth.shape[-1] != 1:
            raise ValueError(f"Expected depth shape (N, H, W, 1), received {tuple(depth.shape)}")

        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(32 * 2 * 2, depth_embedding_dim),
            nn.ELU(),
        )

        proprioception_dim = proprioception.shape[-1]
        network_input_dim = proprioception_dim + depth_embedding_dim
        self.actor = MLP(network_input_dim, num_actions, actor_hidden_dims, activation)
        self.critic = MLP(network_input_dim, 1, critic_hidden_dims, activation)

        self.actor_obs_normalization = actor_obs_normalization
        self.critic_obs_normalization = critic_obs_normalization
        self.actor_obs_normalizer = (
            EmpiricalNormalization(proprioception_dim) if actor_obs_normalization else nn.Identity()
        )
        self.critic_obs_normalizer = (
            EmpiricalNormalization(proprioception_dim) if critic_obs_normalization else nn.Identity()
        )

        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unsupported noise_std_type: {noise_std_type}")

        self.distribution = None
        Normal.set_default_validate_args(False)

        print(f"Depth CNN: {self.depth_encoder}")
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _depth_features(self, obs):
        depth = obs[self.depth_group].permute(0, 3, 1, 2).contiguous()
        return self.depth_encoder(depth)

    def _actor_input(self, obs):
        proprioception = self.actor_obs_normalizer(obs[self.proprioception_group])
        return torch.cat((proprioception, self._depth_features(obs)), dim=-1)

    def _critic_input(self, obs):
        proprioception = self.critic_obs_normalizer(obs[self.proprioception_group])
        return torch.cat((proprioception, self._depth_features(obs)), dim=-1)

    def update_distribution(self, obs):
        mean = self.actor(self._actor_input(obs))
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, obs, **kwargs):
        self.update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        return self.actor(self._actor_input(obs))

    def evaluate(self, obs, **kwargs):
        return self.critic(self._critic_input(obs))

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(obs[self.proprioception_group])
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(obs[self.proprioception_group])

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True


#########################
# RSL-RL registration
#########################


def register_depth_actor_critic():
    """Expose the custom policy to the RSL-RL runner class resolver."""
    import rsl_rl.runners.on_policy_runner as on_policy_runner

    on_policy_runner.DepthActorCritic = DepthActorCritic
