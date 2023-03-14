from config import config
import numpy as np
from pathlib import Path
from plotting import animation_plot
import utils
import signal

class Sarsa:
    def __init__(self, env_name:str):
        self.env_name = env_name
        self.conf = config['sarsa'][env_name]

        self.updater = utils.updaters.get_updater(self.conf)

        # get num actions
        env, _ = utils.get_env(env_name)
        num_actions = env.action_space.n
        env.close()

        self.values = utils.data_structures.ValueMap(num_actions, init_val=self.conf['init_val'])

        self.num_actions = num_actions

        self.fpath = Path("./weights/" + self.conf['save_name'])
    
    def act(self, state):
        if np.random.random() < self.conf['epsilon']:
            return np.random.randint(self.num_actions)
        
        return self.act_optimal(state)

    def act_optimal(self, state):
        return np.argmax(self.values.get(state))

    def num_pairs(self):
        return self.values.num_pairs()

    def load_if_exists(self):
        if self.fpath.exists():
            self.load()

    def load(self):
        self.values.load(self.fpath)

    def save(self):
        self.values.save(self.fpath)

    def learn(self, state, action, reward, next_state=None, next_action=None):

        ## either both should be none or neither should be
        assert(not ((next_state is None) ^ (next_action is None)))

        next_val = 0
        if next_state is not None:
            next_val = self.values.get(next_state)[next_action]

        old_val = self.values.get(state)[action]
        n_visits = self.values.get_visits(state)[action]

        updated = self.updater(old_val, reward + self.conf['gamma'] * next_val, n_visits)

        self.values.set(state, action, updated)
        self.values.update_count(state, action)

    def play(self, load=True):
        env, discretizer = utils.get_env(self.env_name, train=False)

        if load:
            self.load_if_exists()

        s, _ = env.reset()

        while True:
            s_discrete = discretizer(s)
            a = self.act_optimal(s_discrete)

            s, _, done, _, _ = env.step(a)

            if done:
                break

        env.close()



    def train(self, load=True, animate=False):
        
        env, discretizer = utils.get_env(self.env_name, train=True)

        if load:
            self.load_if_exists()


        rewards = []

        upd = None
        if animate:
            upd = animation_plot(rewards)


        def signal_handler(sig, frame):
            self.save()
            env.close()
            exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        
        s, _ = env.reset()
        next_action = self.act(discretizer(s))
        reward = 0

        for i in utils.N():
            
            s_discrete = discretizer(s)

            s, r, done, truncated, info = env.step(next_action)
            reward += r

            if done or truncated:
                self.learn(s_discrete, next_action, r)
            else:
                prev_action = next_action
                s_next_discrete = discretizer(s)
                next_action = self.act(s_next_discrete)
                self.learn(s_discrete, prev_action, r, s_next_discrete, next_action)

            if i % 10000 == 0:
                # upd()
                print(f"Episode: {i}, reward: {r}")
                print(f"Num pairs: {self.num_pairs()}")
                print(s) 
                print(discretizer(s))
                if(len(rewards) > 0):
                    print("Avg episode reward: ", sum(rewards[-1000:]) / len(rewards[-1000:]))
                

            if done or truncated:
                s, _ = env.reset()
                rewards.append(reward)
                reward = 0

        env.close()



