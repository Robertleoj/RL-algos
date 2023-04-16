from .base import AgentBase
import torch
from torch import nn
from collections import deque
import utils
from torch.distributions import Normal



class ACBootstrap(AgentBase):
    agent_name = 'ac_bootstrap'

    class Net(nn.Module):
        def __init__(self, env_name):
            super().__init__()
            self._get_net(env_name)

        def _get_net(self, env_name):
            if env_name == 'Pendulum-v1':

                self.body = nn.Sequential(
                    nn.Linear(3, 64),
                    nn.ReLU(),
                    nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.Linear(64, 64),
                    nn.ReLU()
                )

                self.value_head = nn.Linear(64, 1)

                self.policy_head = nn.Linear(64, 2)


        def forward(self, x):
            # print(x)
            body_out = self.body(x)
            # print(body_out)
            pol_out = self.policy_head(body_out)

            # print(pol_out)
            # apply softplus
            pol_out[1] = torch.log(1 + torch.exp(pol_out[1])) + 1e-4
            # print(pol_out)

            return pol_out, self.value_head(body_out)


    def __init__(self, env_name):
        super().__init__(env_name)
        self.net = self.Net(env_name)

        self.action_tensors = []

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.conf['lr'])

    def act(self, state):
        self.net.train()
        state = torch.tensor(state, dtype=torch.float32)
        
        action_probs, _ = self.net(state)
        mean = action_probs[0]
        std = action_probs[1]
        gaussian = Normal(mean, std)
        action = gaussian.sample()


        # reshape to be 0-dimensional
        self.action_tensors.append(gaussian.log_prob(action))

        return [action.detach().numpy()]

    def act_optimal(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        self.net.eval()
        with torch.no_grad():
            action, _ = self.net(state)
        return action.numpy()

    def learn_episode_end(self, episode_transitions):


        bootstrap_n = self.conf['bootstrap']

        last_n = deque()

        q_vals = []


        value_estimates = []
        for trans in reversed(episode_transitions):
            value_estimates.append(self.net(torch.tensor(trans.state))[1][0])

        for i, trans in enumerate(reversed(episode_transitions)):
            trans: utils.transition

            if len(last_n) < bootstrap_n:
                last_n.appendleft(trans.reward)
            else:
                last_n.appendleft(trans.reward)
                last_n.pop()
                last_n.pop()

                with torch.no_grad():
                    last_n.append(value_estimates[i - bootstrap_n])


            q_vals.append(sum([(self.conf['gamma'] ** i) * r for i, r in enumerate(last_n)]))


        policy_loss = 0
        value_loss = 0

        for q_hat, value, action_log_prob in zip(reversed(q_vals), reversed(value_estimates), self.action_tensors):
            value_loss += (q_hat - value) ** 2
            policy_loss -= (q_hat - value) * action_log_prob

        loss: torch.Tensor = (policy_loss + value_loss) 

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.action_tensors = []

        

