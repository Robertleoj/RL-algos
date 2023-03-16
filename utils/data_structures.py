import numpy as np
import random
import pickle
import heapq
from collections import defaultdict as dd

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

    def zero_out(self, state):
        self.values[state] = np.zeros(self.num_actions)

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

    def update_count(self, state, action, num=1):
        self.assert_state(state)

        self.num_encountered[state][action] += num


class PriorityBuffer:
    def __init__(self):
        self.model = dd(list)
        self.pq = []
        self.rev_map = dd(list)

    def pq_empty(self):
        return len(self.pq) == 0

    def make_key(self, state, action):

        state_key = self.make_state_key(state)

        return (state_key, action)

    def make_state_key(self, state):
        try:
            return tuple(state)
        except TypeError:
            return state

    def insert_model(self, state, action, reward, next_state, done):

        key = self.make_key(state, action)
        self.model[key].append((reward, next_state, done))

        rev_key = self.make_state_key(next_state)
        self.rev_map[rev_key].append((state, action))

    def reverse_sample(self, state):
        key = self.make_state_key(state)
        values = self.rev_map[key]
        choice = random.choice(values)
        return choice

    def sample_model(self, state, action):
        key = self.make_key(state, action)
        values = self.model[key]
        choice= random.choice(values, )
        return choice

    def insert_pq(self, el, priority):
        heapq.heappush(self.pq, (-priority, el))

    def pop_pq(self):
        return heapq.heappop(self.pq)[1]

    def save(self, fpath):
        with open(fpath, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, fpath):
        with open(fpath, 'rb') as f:
            self.buffer = pickle.load(f)

    def pq_size(self):
        return len(self.pq)


class UniformBuffer:
    def __init__(self, buffer_size=None):
        self.buffer_size = buffer_size
        self.buffer = np.array([],dtype=object)

    def __len__(self):
        return len(self.buffer)

    def insert(self, el):
        np.append(self.buffer, np.array([el], dtype=object))

        if self.buffer_size is not None:
            self.buffer = self.buffer[(-self.buffer_size):]

    def save(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load(self, fname):
        with open(fname, 'rb') as f:
            self.buffer = pickle.load(f)

    def sample(self, n):
        return np.random.choice(self.buffer, n)