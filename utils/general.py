import discretizers
import gymnasium as gym
import numpy as np

def N():
    n = 0
    while True:
        yield n
        n += 1

def get_env(env_name, train=True):

    match env_name:
        case 'CartPole-v1':
            discretizer = discretizers.CartPoleDiscretizer()
        case 'MountainCar-v0':
            discretizer = discretizers.MountainCarDiscretizer()

        case 'Acrobot-v1':
            discretizer = discretizers.AcrobotDiscretizer()
    
    if train:
        env = gym.make(env_name)
    else:
        env = gym.make(env_name, render_mode='human')

    return env, discretizer