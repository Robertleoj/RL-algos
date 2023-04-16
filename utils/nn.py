import torch
from torch.distributions import Normal
import numpy as np
import torch.nn.functional as F



def get_action_and_logprobs_normal(mean, log_std, act_limit):
    std = torch.exp(log_std)

    dist = Normal(mean, std)

    samples = dist.rsample()


    logp_pi = dist.log_prob(samples).sum(axis=-1)
    logp_pi -= (2*(np.log(2) - samples - F.softplus(-2*samples))).sum(axis=1)

    samples = torch.tanh(samples)
    samples = samples * act_limit
    # print(samples.shape)
    return samples, logp_pi

def get_action_and_logprobs_categorical(logits):
    dist = torch.distributions.Categorical(logits=logits)
    samples = dist.sample()
    logp_pi = dist.log_prob(samples)
    return samples, logp_pi