import torch
from .base import AgentBase
from .neural_nets import StochasticMultinomialPolicyNet
from collections import deque
from torch.distributions import Categorical
from pathlib import Path
import numpy as np

device='cpu'

class VanillaPolGradient(AgentBase):

    agent_name = 'vanilla_policy_gradient'

    def __init__(self, env_name):
        super().__init__(env_name)
        self.net = StochasticMultinomialPolicyNet(env_name, hidden_shapes=self.conf['hidden_shapes'], device=device).to(device)

        self.batch_size = self.conf['batch_size']
        self.gamma = self.conf['gamma']

        self.replay_buffer = deque()
        self.batch_obs = 0

        self.pol_optimizer = torch.optim.Adam(self.net.parameters(), lr=self.conf['lr'])

    def save(self):
        model_dir = Path(f'./models/{self.agent_name}/{self.env_name}')
        model_dir.mkdir(parents=True, exist_ok=True)

        d = {
            'pol.pt': self.net.state_dict(),
            'pol_opt.pt': self.pol_optimizer.state_dict(),
        }

        for k, v in d.items():
            torch.save(v, model_dir / k)

    def load(self):
        model_dir = Path(f'./models/{self.agent_name}/{self.env_name}')
        model_dir.mkdir(parents=True, exist_ok=True)

        d = {
            'pol.pt': self.net.load_state_dict,
            'pol_opt.pt': self.pol_optimizer.load_state_dict,
        }

        for k, v in d.items():
            v(torch.load(model_dir / k))

    def load_if_exists(self):
        model_dir = Path(f'./models/{self.agent_name}/{self.env_name}')
        if model_dir.exists() and len(list(model_dir.iterdir())) > 0:
            self.load()

    def reset(self):
        model_dir = Path(f'./models/{self.agent_name}/{self.env_name}')

        if model_dir.exists():
            for f in model_dir.iterdir():
                f.unlink()

    def act_optimal(self, state: np.ndarray):
        self.net.eval()
        with torch.no_grad():
            out = self.net(state, just_logits=True)
            return out.argmax().item()
            # return out[0].item()

    def _get_log_prob(self, states, actions):
        logits = self.net(states, just_logits=True)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions)

    def act(self, state):
        self.net.eval()
        with torch.no_grad():
            out = self.net(state)
            return out[0].item()
    
    def learn_episode_end(self, episode_transitions):
        # make batch

        states = torch.tensor([t.state for t in episode_transitions], dtype=torch.float32).to(device)
        # print([t.action for t in episode_transitions])
        actions = torch.tensor([t.action for t in episode_transitions], dtype=torch.int64).to(device)
        R = sum([t.reward * (self.gamma ** i) for i, t in enumerate(episode_transitions)]) / self.conf['reward_div']

        self.replay_buffer.append((states, actions, R))

        self.batch_obs += states.shape[0]

        if self.batch_obs >= self.batch_size:
            self._train_batch()
            self.replay_buffer.clear()
            self.batch_obs = 0

    def _train_batch(self):
        print('training batch')
        self.net.train()
    
        loss = 0

        for states, actions, R in self.replay_buffer:
            log_prob = self._get_log_prob(states, actions)
            loss -= log_prob.sum() * R

        loss /= len(self.replay_buffer)

        self.pol_optimizer.zero_grad()
        loss.backward()
        self.pol_optimizer.step()

