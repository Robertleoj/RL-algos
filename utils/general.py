import discretizers
import gymnasium as gym
import numpy as np
import flappy_bird_gym


def N():
    n = 0
    while True:
        yield n
        n += 1

def get_env(env_name, train=True, cont=False):

    if env_name == "FlappyBird-v0":
        env = flappy_bird_gym.make("FlappyBird-v0")
        return env
    if env_name == "LunarLander-v2":
        if train:
            env = gym.make(env_name, continuous=cont)
        else:
            env = gym.make(env_name, continuous=cont, render_mode='human')
        return env
    

    if train:
        env = gym.make(env_name)
    else:
        env = gym.make(env_name, render_mode='human')

    return env

