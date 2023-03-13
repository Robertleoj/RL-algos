import numpy as np
from .utils import AvgUpdater, ValueMap

EPSILON = 0.1
GAMMA = 0.9

class QAgentVanilla:
    def __init__(self, num_actions):
        self.updater = AvgUpdater()
        self.values = ValueMap(num_actions)
        self.num_actions = num_actions

    def act(self, state):

        if np.random.random() < EPSILON:
            return np.random.randint(self.num_actions)

        return np.argmax(self.values.get(state))

    def num_pairs(self):
        return self.values.num_pairs()


    def learn(self, state, action, reward, next_state=None):


        next_val = 0
        if next_state is not None:
            next_val = np.max(self.values.get(next_state))

        old_val = self.values.get(state)[action]
        n_visits = self.values.get_visits(state)[action]

        updated = self.updater(old_val, reward + GAMMA * next_val, n_visits)


        self.values.set(state, action, updated)
        self.values.update_count(state, action)





