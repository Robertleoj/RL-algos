import numpy as np
import pickle
from config import config
from pathlib import Path
import utils
from plotting import animation_plot
import gymnasium as gym
import signal


class DynaQ:
    def __init__(self, env_name):

        self.env_name = env_name

        self.conf = config['DynaQ'][env_name]

        self.updater = utils.updaters.get_updater(self.conf)

        # get num actions
        env, _ = utils.get_env(env_name)
        num_actions = env.action_space.n
        env.close()

        self.values = utils.data_structures.ValueMap(num_actions, init_val=self.conf['init_val'])

        self.buffer = utils.data_structures.DynaBuffer(self)

        self.num_actions = num_actions

        self.fpath = Path("./weights/" + self.conf['save_name'])
        self.buffer_path = Path("./buffers/" + self.conf['buffer_save_name'])

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

        if self.buffer_path.exists():
            self.buffer.load(self.buffer_path)

    def save(self):
        self.values.save(self.fpath)
        self.buffer.save(self.buffer_path)

    def load(self):
        self.values.load(self.fpath)
        self.buffer.load(self.buffer_path)

    def learn(self, state, action, reward, next_state=None, update_count = True):


        next_val = 0
        if next_state is not None:
            next_val = np.max(self.values.get(next_state))

        old_val = self.values.get(state)[action]
        n_visits = self.values.get_visits(state)[action]

        updated = self.updater(old_val, reward + self.conf['gamma'] * next_val, n_visits)


        self.values.set(state, action, updated)

        if update_count:
            self.values.update_count(state, action)

    def plan(self):
        if len(self.buffer) < self.conf['n_planning_steps']:
            return

        samples = self.buffer.sample(self.conf['n_planning_steps'])

        for sample in samples:
            s, a, r, s_next = sample
            self.learn(s, a, r, s_next, update_count=False)

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

        s, _ = env.reset()

        rewards = []

        upd = None
        if animate:
            upd = animation_plot(rewards)

        reward = 0

        def signal_handler(sig, frame):
            self.save()
            env.close()
            exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        n_episodes = 0

        for i in utils.N():
            
            s_discrete = discretizer(s)
            a = self.act(s_discrete)

            s, r, done, truncated, info = env.step(a)
            reward += r

            if done or truncated:
                self.learn(s_discrete, a, r, None)
            else:
                s_next_discrete = discretizer(s)
                self.learn(s_discrete, a, r, s_next_discrete)

            if i % 10000 == 0:
                # upd()
                print(f"Episode: {n_episodes}, reward: {r}")
                print(f"Num pairs: {self.num_pairs()}")
                print(s) 
                print(discretizer(s))
                if(len(rewards) > 0):
                    print("Avg ep reward: ", sum(rewards[-1000:]) / len(rewards[-1000:]))
                

            if done or truncated:
                s, _ = env.reset()
                rewards.append(reward)
                reward = 0
                n_episodes += 1

            self.plan()

        env.close()




