from .base import AgentBase
import torch
from torch import nn
import numpy as np
from collections import deque
import utils
import itertools
from torch.distributions import Normal
import random
from pathlib import Path
import torch.nn.functional as F
from preprocessors import get_preprocessor

device = 'cpu'

LOG_STD_MAX = 2
LOG_STD_MIN = -20

def tofloattensor(x):
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)
    else:
        x = x.to(torch.float32)
    return x

class PolicyNet(nn.Module):
    def __init__(self, env_name):
        super().__init__()
        self.env_name = env_name
        self.preprocessor = get_preprocessor(env_name)
        self._get_net()

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

        
        x = tofloattensor(x).to(device)
        # print(x.shape, x.dtype)

        pol_out = self.net(x)
        mean = self.mean(pol_out)
        if deterministic:
            return mean

        log_std = self.logstd(pol_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)


        std = torch.exp(log_std)

        dist = Normal(mean, std)

        samples = dist.rsample()


        logp_pi = dist.log_prob(samples).sum(axis=-1)
        logp_pi -= (2*(np.log(2) - samples - F.softplus(-2*samples))).sum(axis=1)

        samples = torch.tanh(samples)
        samples = samples * self.act_limit
        # print(samples.shape)
        return samples, logp_pi

class QNet(nn.Module):
    def __init__(self, env_name):
        super().__init__()
        self._get_net(env_name)
        self.preprocessor = get_preprocessor(env_name)
    
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

        s = tofloattensor(s).to(device)
        a = tofloattensor(a).to(device)

        inp = torch.cat((s, a), dim=1)
        return self.net(inp)


class SAC(AgentBase):
    agent_name = 'SAC'


    def __init__(self, env_name):
        super().__init__(env_name)

        self.pol_net = PolicyNet(env_name).to(device)
        self.q1 = QNet(env_name).to(device)
        self.q2 = QNet(env_name).to(device)
        self.q1_targ = QNet(env_name).to(device)
        self.q2_targ = QNet(env_name).to(device)

        self.q1_targ.load_state_dict(self.q1.state_dict())
        self.q2_targ.load_state_dict(self.q2.state_dict())



        self.replay_buffer = deque(maxlen=self.conf['buffer_size'])
        self.since_last_sampled = 0
        self.total_steps = 0

        self.pol_optimizer = torch.optim.Adam(self.pol_net.parameters(), lr=self.conf['lr'])
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=self.conf['lr'])
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=self.conf['lr'])

    def save(self):
        model_dir = Path(f'./models/{self.agent_name}/{self.env_name}')
        model_dir.mkdir(parents=True, exist_ok=True)

        d = {
            'q1.pt': self.q1.state_dict(),
            'q1_opt.pt': self.q1_optimizer.state_dict(),

            'q2.pt': self.q2.state_dict(),
            'q2_opt.pt': self.q2_optimizer.state_dict(),

            'pol.pt': self.pol_net.state_dict(),
            'pol_opt.pt': self.pol_optimizer.state_dict(),

            'q1_t.pt': self.q1_targ.state_dict(),
            'q2_t.pt': self.q2_targ.state_dict(),
            'data.pt': (self.replay_buffer, self.since_last_sampled, self.total_steps),
        }

        for name, state in d.items():
            torch.save(state, model_dir / name)


    def reset(self):
        # delete all saved data
        model_dir = Path(f'./models/{self.agent_name}/{self.env_name}')
        if not model_dir.exists():
            return
        for file in model_dir.iterdir():
            file.unlink()


    def load(self):
        self.__init__(self.env_name)

    
        model_dir = Path(f'./models/{self.agent_name}/{self.env_name}')

        self.q1.load_state_dict(torch.load( model_dir / 'q1.pt'))
        self.q1_targ.load_state_dict(torch.load(model_dir / 'q1_t.pt'))
        self.q1_optimizer.load_state_dict(torch.load(model_dir / 'q1_opt.pt'))
        # for param in self.q1_optimizer.param_groups:
        #     param['lr'] = self.conf['lr']


        self.q2.load_state_dict(torch.load(model_dir / 'q2.pt'))
        self.q2_targ.load_state_dict(torch.load(model_dir / 'q2_t.pt'))
        self.q2_optimizer.load_state_dict(torch.load(model_dir / 'q2_opt.pt'))
        # for param in self.q2_optimizer.param_groups:
        #     param['lr'] = self.conf['lr']

        self.pol_net.load_state_dict(torch.load(model_dir / 'pol.pt'))
        self.pol_optimizer.load_state_dict(torch.load(model_dir / 'pol_opt.pt'))
        # for param in self.pol_optimizer.param_groups:
        #     param['lr'] = self.conf['lr']



        data = torch.load(model_dir / 'data.pt')
        self.replay_buffer, self.since_last_sampled, self.total_steps = data

    def load_if_exists(self):
        model_dir = Path(f'./models/{self.agent_name}/{self.env_name}')

        if model_dir.exists() and len(list(model_dir.iterdir())) > 0:
            self.load()


    def act(self, state):
        state_tensor = torch.tensor(state).unsqueeze(0).to(device)
        self.pol_net.eval()
        with torch.no_grad():
            return self.pol_net(state_tensor)[0][0].cpu().numpy()

    def act_random(self):
        if self.total_steps < self.conf['start_steps']:
            return True

    def act_optimal(self, state: np.ndarray):
        # return self.act(state)

        with torch.no_grad():
            state_tensor = torch.tensor(state).unsqueeze(0).to(device)
            # return self.pol_net(state_tensor)[0][0].numpy()
            return self.pol_net(state_tensor, deterministic=True)[0].cpu().numpy()

    def process_transition(self, transition: utils.transition):
        # print(transition)
        transition.reward = transition.reward 
        self.replay_buffer.append(transition)
        self.total_steps += 1

    def learn(self, episode_transition): 
        self.since_last_sampled += 1
    
        if self.since_last_sampled % self.conf['update_rate'] == 0:
            self.since_last_sampled = 0
            self._train()
        
    def _q_update(self, reward_tensor, state_tensor, old_state_tensor, action_tensor, done_tensor):
        # compute q targets
        with torch.no_grad():
            sampled_actions, log_probs = self.pol_net(state_tensor)
            log_probs = log_probs.unsqueeze(1)

            q1_out = self.q1_targ(state_tensor, sampled_actions)
            q2_out = self.q2_targ(state_tensor, sampled_actions)

            q_min = torch.min(q1_out, q2_out)


        q_targ = (reward_tensor
            + self.conf['gamma'] 
            * (1 - done_tensor) 
            * (q_min - self.conf['entropy_alpha'] * log_probs)
        )

        q1_out = self.q1(old_state_tensor, action_tensor)
        q2_out = self.q2(old_state_tensor, action_tensor)

        mse = torch.nn.MSELoss()

        q1_loss = mse(q1_out, q_targ)
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        self.q1_optimizer.zero_grad()


        q2_loss = mse(q2_out, q_targ)

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        self.q2_optimizer.zero_grad()


    def _update(self, samples):

        reward_tensor = torch.tensor(np.array([s.reward for s in samples]), dtype=torch.float32).unsqueeze(1).to(device)
        action_tensor = torch.tensor(np.array([s.action for s in samples]), dtype=torch.float32).to(device)
        old_state_tensor = torch.tensor(np.array([s.state for s in samples]), dtype=torch.float32).to(device)
        state_tensor = torch.tensor(np.array([s.next_state for s in samples]), dtype=torch.float32).to(device)
        done_tensor = torch.tensor(np.array([s.done for s in samples]), dtype=torch.int32).unsqueeze(1).to(device)

        self._q_update(reward_tensor, state_tensor, old_state_tensor, action_tensor, done_tensor)

        for p in itertools.chain(self.q1.parameters(), self.q2.parameters()):
            p.requires_grad = False

        # policy update
        sampled_actions, log_probs = self.pol_net(old_state_tensor)

        q_min = torch.min(
            self.q1(old_state_tensor, sampled_actions), 
            self.q2(old_state_tensor, sampled_actions)
        )

        pol_loss = (self.conf['entropy_alpha'] * log_probs - q_min).mean()
        self.pol_optimizer.zero_grad()
        pol_loss.backward()
        self.pol_optimizer.step()
        self.pol_optimizer.zero_grad()

        for p in itertools.chain(self.q1.parameters(), self.q2.parameters()):
            p.requires_grad = True



        # update target distributions
        rho = self.conf['update_rho']
        for q, q_targ in zip(
            itertools.chain(self.q1.parameters(), self.q2.parameters()),
            itertools.chain(self.q1_targ.parameters(), self.q2_targ.parameters())
        ):

            q_targ.data.mul_(1 - rho)
            q_targ.data.add_((rho) * q.data)


    def _train(self):

        self.pol_net.train()


        if len(self.replay_buffer) < self.conf['batch_size'] or self.total_steps < self.conf['start_steps']:
            return

        for _ in range(self.conf['num_updates']):
            samples = random.sample(self.replay_buffer, self.conf['batch_size'])
            self._update(samples)








    
