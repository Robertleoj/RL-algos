import gymnasium as gym
from discretizers import CartPoleDiscretizer
from agents import QAgentVanilla

env = gym.make('CartPole-v1', render_mode="human")
agent = QAgentVanilla(env.action_space.n)

s, info = env.reset()

discretizer = CartPoleDiscretizer()

lengths = []

ep_length = 0
for i in range(100000000):
    
    s_discrete = discretizer(s)
    a = agent.act(s_discrete)
    s, r, done, truncated, info = env.step(a)
    ep_length += 1
    

    if done:
        agent.learn(s_discrete, a, r, None)
    else:
        s_next_discrete = discretizer(s)
        agent.learn(s_discrete, a, r, s_next_discrete)

    if i % 100 == 0:
        print(f"Episode: {i}, reward: {r}")
        print(f"Num pairs: {agent.num_pairs()}")
        if(len(lengths) > 0):
            print("Avg length: ", sum(lengths) / len(lengths))
        


    if done:
        s, info = env.reset()
        lengths.append(ep_length)
        ep_length = 0
        lengths = lengths[-100:]


env.close()
