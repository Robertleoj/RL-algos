import numpy as np

def N():
    n = 0
    while True:
        yield n
        n += 1

class LRUpdater:
    def __init__(self, lr, lr_decay):
        self.lr = lr
        self.lr_decay = lr_decay

    def __call__(self, curr_val, new_val, num_encounters):
        lr = self.lr * (1 / (1 + self.lr_decay * num_encounters))

        return curr_val + lr * (new_val - curr_val)

class AvgUpdater:

    def __call__(self, curr_val, new_val, num_encounters):
        return (num_encounters * curr_val + new_val) / (num_encounters + 1)
    

class ValueMap:
    def __init__(self, num_actions):
        self.values = {}
        self.num_encountered = {}
        self.num_actions = num_actions

    def assert_state(self, state):
        if state not in self.values:
            self.values[state] = np.zeros(self.num_actions)

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


