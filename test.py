import agents
import gymnasium as gym


# agent = agents.DynaQ("Acrobot-v1")
# agent = agents.DynaQ("FlappyBird-v0")
# agent = agents.DynaQPrioritized("Acrobot-v1")
# agent = agents.DynaQPrioritized("FlappyBird-v0")
# agent = agents.MonteCarlo("Taxi-v3")
# agent = agents.NStepSarsa("CartPole-v1")
agent = agents.NStepSarsa("FlappyBird-v0")
# agent = agents.NStepSarsa("Acrobot-v1")
# agent = agents.Sarsa("CartPole-v1")
# agent = agents.Sarsa("Taxi-v3")
# agent = agents.QAgent("Taxi-v3")
# agent = agents.Sarsa("MountainCar-v0")

agent.play(20)

# agent.reset()
# agent.train()

# fl = gym.make('FlappyBird-v0')
# print(fl.observation_space)

# fl.reset()

# mn = float('inf')
# mx = - float('inf')


# while True:
#     fl.render()
#     s, _, done, _, _ = fl.step(fl.action_space.sample())
#     print(s)

#     mn = min(mn, s[1])
#     mx = max(mx, s[1])

#     print(done)

#     if done:
#         fl.reset()

#     print(mn, mx)


