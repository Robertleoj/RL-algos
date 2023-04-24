from torch import nn
import torch
import gymnasium as gym
import utils
from config import config
from .base import AgentBase




class PPO(AgentBase):
    def __init__(self, env_name):
        self.env_name = env_name
        self.conf = config['PPO'][env_name]
        self._init_networks(self)

    def _init_networks(self):

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

            self.pol_net = utils.neural.StochasticNormalPolicyNet(
                obs_space, 
                action_space,
                self.conf['pol_hidden_shapes'],
                act_limit = action_bounds
            )

        # create the critic network, which is a QNet
        self.critic_net = utils.neural.ValueNet(
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
