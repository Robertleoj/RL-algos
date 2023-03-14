import gymnasium as gym
from discretizers import CartPoleDiscretizer
import agents


agent = agents.DynaQ("Acrobot-v1")
# agent = agents.Sarsa("CartPole-v1")

agent.train()
