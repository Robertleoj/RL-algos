import numpy as np
import pickle

class ValueMap:
    def __init__(self, num_actions, init_val=0):
        self.values = {}
        self.num_encountered = {}
        self.num_actions = num_actions
        self.init_val = init_val

    def save(self, path):
        data = (self.values, self.num_encountered)
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.values, self.num_encountered = pickle.load(f)

    def assert_state(self, state):
        if state not in self.values:
            self.values[state] = np.zeros(self.num_actions) + self.init_val

        if state not in self.num_encountered:
            self.num_encountered[state] = np.zeros(self.num_actions) 

    def get(self, state):
        self.assert_state(state)
        return self.values[state]

    def get_visits(self, state):
        self.assert_state(state)
        return self.num_encountered[state]

    def num_pairs(self):
        return len(self.values) * self.num_actions

    def set(self, state, action, val):
        self.assert_state(state)

        self.values[state][action] = val

    def update_count(self, state, action):
        self.assert_state(state)

        self.num_encountered[state][action] += 1


class DynaBuffer:
    def __init__(self, buffer_size=None):
        self.buffer_size = buffer_size
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def insert(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

        if self.buffer_size is not None:
            self.buffer = self.buffer[-self.buffer_size:]

    def save(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load(self, fname):
        with open(fname, 'rb') as f:
            self.buffer = pickle.load(f)

    def sample(self, n):
        return np.random.choice(self.buffer, n)