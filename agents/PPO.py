from torch import nn
import torch
import gymnasium as gym
from .neural import ValueNet, StochasticMultinomialPolicyNet, StochasticNormalPolicyNet
from config import config
from .base import AgentBase


class PPO(AgentBase):
    def __init__(self, env_name):
        self.env_name = env_name
        self.conf = config['PPO'][env_name]
        self._init_networks(self)

    def _init_networks(self):

        env = gym.make(self.env_name)

        # get the observation space shape
        obs_space = env.observation_space.shape[0]

        # check whether the environment is discrete or continuous
        is_discrete = isinstance(env.action_space, gym.spaces.Discrete)

        # if discrete, use a multinomial policy net
        if is_discrete:
            # get the number of actions
            action_space = env.action_space.n

            self.actor_net = StochasticMultinomialPolicyNet(
                obs_space, 
                action_space,
                self.conf['pol_hidden_shapes'],
            )   

        else:
            # get the number of actions
            action_space = env.action_space.shape[0]

            # get the action space bounds, low and high
            action_space_low = env.action_space.low
            action_space_high = env.action_space.high

            action_bounds = list(zip(action_space_low, action_space_high))

            self.pol_net = StochasticNormalPolicyNet(
                obs_space, 
                action_space,
                self.conf['pol_hidden_shapes'],
                act_limit = action_bounds
            )

        # create the critic network, which is a QNet
        self.critic_net = ValueNet(
            obs_space,
            self.conf['critic_hidden_shapes']
        )


    def play():
        super().play()
        pass

    def train():
        super().train()
        pass

    def reset():
        pass

    def save():
        pass

    def load():
        pass
