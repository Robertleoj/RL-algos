import gymnasium as gym
from discretizers import CartPoleDiscretizer
from agents import QAgent

TRAIN = True

if TRAIN:
    env = gym.make('CartPole-v1')
else:
    env = gym.make('CartPole-v1', render_mode='human')

discretizer = CartPoleDiscretizer()

agent = QAgent(env.action_space.n)

agent.train(env, discretizer, load=True, animate=False)

