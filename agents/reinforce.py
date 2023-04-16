from .base import AgentBase
import torch
from torch import nn
from collections import deque
import utils



class Reinforce(AgentBase):
    agent_name = 'reinforce'

    class Net(nn.Module):
        def __init__(self, env_name):
            super().__init__()
            self._get_net(env_name)

        def _get_net(self, env_name):
            if env_name == 'Pendulum-v1':
                self.net = nn.Sequential(
                    nn.Linear(3, 64),
                    nn.ReLU(),
                    nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                )

        def forward(self, x):
            return self.net(x)


    def __init__(self, env_name):
        super().__init__(env_name)
        self.net = self.Net(env_name)
        self.action_tensors = []
        self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=self.conf['lr'])

    def act(self, state):
        self.net.train()
        state = torch.tensor(state, dtype=torch.float32)
        action = self.net(state)
        # reshape to be 0-dimensional
        self.action_tensors.append(action[0])

        return action.detach().numpy()

    def act_optimal(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        self.net.eval()
        with torch.no_grad():
            action = self.net(state)
        return action.numpy()

    def learn_episode_end(self, episode_transitions):
        R = 0

        loss = torch.tensor(0, dtype=torch.float32)

        returns = []
        for trans in reversed(episode_transitions):
            R = trans.reward  + self.conf['gamma'] * R
            returns.append(R)

        returns = torch.tensor(returns[::-1], dtype=torch.float32)

        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        # for trans in reversed(episode_transitions):
        #     trans: utils.transition

        #     R = trans.reward  + self.conf['gamma'] * R

        #     loss -= R * torch.log(self.action_tensors.pop())

        for R, action in zip(returns, self.action_tensors):
            loss -= R * torch.log(action)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.action_tensors = []

        



