import numpy as np
import pickle
from config import config
from pathlib import Path
from .utils import AvgUpdater, ValueMap, N
from plotting import animation_plot
import signal

conf = config['q_learning']

class QAgent:
    def __init__(self, num_actions):
        self.updater = AvgUpdater()
        self.values = ValueMap(num_actions)
        self.num_actions = num_actions
        self.fpath = Path("./weights/" + conf['save_name'])

    def act(self, state):

        if np.random.random() < conf['epsilon']:
            return np.random.randint(self.num_actions)

        return self.act_optimal(state)

    def act_optimal(self, state):
        return np.argmax(self.values.get(state))

    def num_pairs(self):
        return self.values.num_pairs()

    def load_if_exists(self):
        if self.fpath.exists():
            self.load()

    def save(self):
        with open(self.fpath, 'wb') as f:
            pickle.dump(self.values, f)

    def load(self):
        if not self.fpath.exists():
            raise ValueError(f'No save found')

        with open(self.fpath, 'rb') as f:
            self.values = pickle.load(f)

    def learn(self, state, action, reward, next_state=None):


        next_val = 0
        if next_state is not None:
            next_val = np.max(self.values.get(next_state))

        old_val = self.values.get(state)[action]
        n_visits = self.values.get_visits(state)[action]

        updated = self.updater(old_val, reward + conf['gamma'] * next_val, n_visits)


        self.values.set(state, action, updated)
        self.values.update_count(state, action)

    def play(self, env, discretizer, load=True, animate=False):
        if load:
            self.load_if_exists()

        s, _ = env.reset()

        if animate:
            env.render()

        while True:
            s_discrete = discretizer(s)
            a = self.act_optimal(s_discrete)

            s, _, done, _, _ = env.step(a)

            if animate:
                env.render()

            if done:
                break

        env.close()

    def train(self, env, discretizer, load=True, animate=False):


        if load:
            self.load_if_exists()

        s, _ = env.reset()

        lengths = []

        upd = None
        if animate:
            upd = animation_plot(lengths)

        ep_length = 0

        def signal_handler(sig, frame):
            self.save()
            env.close()
            exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        

        for i in N():
            
            s_discrete = discretizer(s)
            a = self.act(s_discrete)

            s, r, done, truncated, info = env.step(a)
            ep_length += 1

            if done:
                self.learn(s_discrete, a, r, None)
            else:
                s_next_discrete = discretizer(s)
                self.learn(s_discrete, a, r, s_next_discrete)

            if i % 10000 == 0:
                # upd()
                print(f"Episode: {i}, reward: {r}")
                print(f"Num pairs: {self.num_pairs()}")
                print(s) 
                print(discretizer(s))
                if(len(lengths) > 0):
                    print("Avg length: ", sum(lengths[-1000:]) / len(lengths[-1000:]))
                

            if done:
                s, _ = env.reset()
                lengths.append(ep_length)
                ep_length = 0

        env.close()




