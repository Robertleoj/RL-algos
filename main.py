import agents

agent = agents.Reinforce('CartPole-v0')

agent.train()

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


