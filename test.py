import agents


# agent = agents.DynaQ("Acrobot-v1")
# agent = agents.MonteCarlo("Taxi-v3")
agent = agents.OffPolicyMonteCarlo("CartPole-v1")
# agent = agents.Sarsa("CartPole-v1")
# agent = agents.Sarsa("Taxi-v3")
# agent = agents.QAgent("Taxi-v3")
# agent = agents.Sarsa("MountainCar-v0")

# agent.play(20)

agent.reset()
agent.train()
