from .base import AgentBase
import gymnasium as gym
import torch
from pathlib import Path
import utils




class Reinforce(AgentBase):
    """
    Reinforce agent - uses a policy gradient method to learn a policy
    """

    agent_name = 'reinforce'

    def __init__(self, env_name):
        super().__init__(env_name)

        self._make_actor()
        self._make_save_paths()

    def _make_save_paths(self):
        self.save_dir = Path('saves') / self.agent_name / self.env_name

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.save_dir / 'model.pt'

    def save(self):

        torch.save(self.actor_net.state_dict(), self.model_path)

    def load(self):
        if self.model_path.exists():
            self.actor_net.load_state_dict(torch.load(self.model_path))

    def _make_actor(self):
        env = gym.make(self.env_name)

        is_discrete, obs_space, action_space, action_bounds = utils.env_info(env)

        # if discrete, use a multinomial policy net
        if is_discrete:

            self.actor_net = utils.neural.StochasticMultinomialPolicyNet(
                obs_space, 
                action_space,
                self.conf['pol_hidden_shapes'],
            )   

        else:

            self.actor_net = utils.neural.StochasticNormalPolicyNet(
                obs_space, 
                action_space,
                self.conf['pol_hidden_shapes'],
                act_limit = action_bounds
            )

    def act(self, s):
        s = torch.tensor(s).unsqueeze(0).float()
        self.actor_net.eval()
        with torch.no_grad():
            a, _ = self.actor_net(s)

            return a.numpy()[0]

    def play(self, load=True):
        super().play(load)

        env = gym.make(self.env_name, render=True)

        s, done = env.reset()

        while True:
            a = self.act(s)

            s, r, done, trunc, info = env.step(a)

            if done or trunc:
                s, done = env.reset()

    def train(self, load=True):
        super().train(load)

        # create the optimizer
        optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=self.conf['lr'])

        num_transitions = 0
        num_episodes = 0

        log_interval = 10

        buffer = []
        trajectory = []

        env = gym.make(self.env_name)

        s, done = env.reset()
        trajectory.append(s)

        episode_rewards = []
        rewards = []

        while True:

            a = self.act(s)

            s, r, done, trunc, info = env.step(a)
            rewards.append(r)

            trajectory.extend([a, r, s])

            num_transitions+= 1

            if done or trunc:

                episode_rewards.append(sum(rewards))
                rewards.clear()
                num_episodes += 1

                buffer.append(trajectory)
                if num_transitions > self.conf['batch_size']:
                    self._train(buffer, optimizer)
                    buffer.clear()
                    num_transitions = 0
                
                trajectory.clear()

                if num_episodes % log_interval == 0:
                    
                    print(f'Episode {num_episodes} - {sum(episode_rewards)/len(episode_rewards)}')
                    episode_rewards.clear()

                s, done = env.reset()


    def _train(self, trajectories, optimizer):
        """
        train the actor network using the REINFORCE algorithm
        """

        print(trajectories)

        self.actor_net.train()

        states = []
        actions = []
        returns = []

        for trajectory in trajectories:

            # get the states
            t_states = trajectory[::3]

            # get the actions
            t_actions = trajectory[1::3]

            # get the rewards
            t_rewards = trajectory[2::3]

            # calculate the returns
            ret = []
            g = 0

            for r in reversed(t_rewards):
                g = r + self.conf['gamma'] * g
                ret.append(g)
            ret = list(reversed(ret))

            for s, a, r in zip(t_states, t_actions, ret):
                states.append(s)
                actions.append(a)
                returns.append(r)

        states = torch.tensor(states).float()
        actions = torch.tensor(actions).float()
        returns = torch.tensor(returns).float()

        print(states.shape)
        log_probs = self.actor_net.log_probs(states, actions)
        loss = (-log_probs * returns).sum() / len(trajectories)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        