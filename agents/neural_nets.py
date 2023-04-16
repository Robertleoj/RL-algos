import torch
from torch import nn
import  numpy as np
from preprocessors import get_preprocessor
import utils
import gymnasium as gym

LOG_STD_MAX = 2
LOG_STD_MIN = -20


def tofloattensor(x):
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)
    else:
        x = x.to(torch.float32)
    return x


class StochasticMultinomialPolicyNet(nn.Module):
    def __init__(self, env_name, hidden_shapes=[32], device='cpu'):
        super().__init__()
        self.preprocessor = get_preprocessor(env_name)
        self.env_name = env_name
        self.hidden_shapes = hidden_shapes
        self.tensor_device = device
        self._get_net()

    def _get_net(self):
        env = gym.make(self.env_name)
        action_out = env.action_space.n
        space_in = env.observation_space.shape[0]

        in_layer = nn.Linear(space_in, self.hidden_shapes[0])
        hidden_layers = [nn.Linear(self.hidden_shapes[i], self.hidden_shapes[i + 1]) for i in range(len(self.hidden_shapes) - 1)]
        out_layer = nn.Linear(self.hidden_shapes[-1], action_out)

        self.net = nn.Sequential(
            in_layer, *hidden_layers, out_layer
        )

    def forward(self, x, just_logits=False):
        if self.preprocessor is not None:
            x = self.preprocessor(x)

        x = tofloattensor(x)

        if just_logits:
            return self.net(x)
        else:
            logits = self.net(x)
            return utils.get_action_and_logprobs_categorical(logits)

class StochasticNormalPolicyNet(nn.Module):
    def __init__(self, env_name, device='cpu'):
        super().__init__()
        self.env_name = env_name
        self.preprocessor = get_preprocessor(env_name)
        self._get_net()
        self.tensor_device = device

    def _get_net(self):
        if self.env_name == 'Pendulum-v1':

            self.net = nn.Sequential(
                nn.Linear(3, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU()
            )

            self.mean = nn.Linear(64, 1)
            self.logstd = nn.Linear(64, 1)

            self.act_limit = 2

        elif self.env_name == "LunarLander-v2":
            self.net = nn.Sequential(
                nn.Linear(8, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU()
            )

            self.mean = nn.Linear(64, 2)
            self.logstd = nn.Linear(64, 2)

            self.act_limit = 1


    def forward(self, x, deterministic=False):
        if self.preprocessor is not None:
            x = self.preprocessor(x.cpu())

        
        x = tofloattensor(x).to(self.tensor_device)
        # print(x.shape, x.dtype)

        pol_out = self.net(x)
        mean = self.mean(pol_out)
        if deterministic:
            return mean

        log_std = self.logstd(pol_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)


        return utils.get_action_and_logprobs_normal(mean, log_std, self.act_limit)


class QNet(nn.Module):
    def __init__(self, env_name, device):
        super().__init__()
        self._get_net(env_name)
        self.preprocessor = get_preprocessor(env_name)
        self.tensor_device = device
    
    def _get_net(self, env_name):
        if env_name == 'Pendulum-v1':
            self.net = nn.Sequential(
                nn.Linear(4, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
        elif env_name == 'LunarLander-v2':
            self.net = nn.Sequential(
                nn.Linear(10, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )

    def forward(self, s, a):

        if self.preprocessor is not None:
            s = self.preprocessor(s.cpu())

        s = tofloattensor(s).to(self.tensor_device)
        a = tofloattensor(a).to(self.tensor_device)

        inp = torch.cat((s, a), dim=1)
        return self.net(inp)

