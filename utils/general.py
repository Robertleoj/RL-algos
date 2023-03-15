import discretizers
import gymnasium as gym
import numpy as np

def N():
    n = 0
    while True:
        yield n
        n += 1

def get_env(env_name, train=True):


    if train:
        env = gym.make(env_name)
    else:
        env = gym.make(env_name, render_mode='human')

    return env

