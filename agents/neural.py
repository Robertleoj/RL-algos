import torch
from torch import nn
import gymnasium as gym
from torch.distributions import Normal, Categorical
import numpy as np
import torch.nn.functional as F

def mlp(sizes, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = nn.ReLU if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class QNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[64, 64]):
        super().__init__()
        self.net = mlp([state_dim + action_dim] + hidden_dims + [1])

    def forward(self, state: torch.Tensor, action: torch.Tensor):

        assert isinstance(state, torch.Tensor)
        assert isinstance(action, torch.Tensor)

        inp = torch.cat([state, action], dim=-1)
        return self.net(inp)

class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dims=[64, 64]):
        super().__init__()
        self.net = mlp([state_dim] + hidden_dims + [1])

    def forward(self, state: torch.Tensor):
        assert isinstance(state, torch.Tensor)
        return self.net(state)

class StochasticMultinomialPolicyNet(nn.Module):

    def __init__(self, shape_in, action_shape, hidden_shapes=[64, 64]):
        super().__init__()
        self.middle = mlp([shape_in] + hidden_shapes + [action_shape])


    def get_action_and_logprobs_categorical(self, logits):
        dist = torch.distributions.Categorical(logits=logits)
        samples = dist.sample()
        logp_pi = dist.log_prob(samples)
        return samples, logp_pi

    def forward(self, x, just_logits=False):
        assert isinstance(x, torch.Tensor)

        if just_logits:
            return self.net(x)
        else:
            logits = self.net(x)
            return self.get_action_and_logprobs_categorical(logits)

class StochasticNormalPolicyNet(nn.Module):
    def __init__(self, shape_in, action_shape, hidden_shapes=[64, 64], act_limit=1):
        super().__init__()
        self.act_limit = act_limit
        self.middle = mlp([shape_in] + hidden_shapes, output_activation=nn.ReLU())
        self.mean_head = nn.Linear(hidden_shapes[-1], action_shape)
        self.logstd_head = nn.Linear(hidden_shapes[-1], action_shape)

    def get_action_and_logprobs_normal(self, mean, log_std):

        std = torch.exp(log_std)
        dist = Normal(mean, std)

        samples = dist.rsample()

        logp_pi = dist.log_prob(samples).sum(axis=-1)
        logp_pi -= (2*(np.log(2) - samples - F.softplus(-2*samples))).sum(axis=1)

        samples = torch.tanh(samples)

        # check whether act_limit is a number
        if isinstance(self.act_limit, (int, float)):
            samples = samples * self.act_limit
        else:
            modified_samples = torch.zeros_like(samples)
            for i, (a, b) in enumerate(self.act_limit):
                modified_samples[:, i] = samples[:, i] * (b - a) / 2 + (b + a) / 2
            samples = modified_samples

        return samples, logp_pi



    def forward(self, x, deterministic=False):

        middle_out = self.middle(x)
        mean = self.mean_head(middle_out)

        if deterministic:
            return mean

        log_std = self.logstd_head(middle_out)

        return self.get_action_and_logprobs_normal(mean, log_std)

