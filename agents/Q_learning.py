import numpy as np
from pathlib import Path
import utils
import discretizers
from .base import AgentBase
from os import remove


class QAgent(AgentBase):
    agent_name = 'q_learning'

    def __init__(self, env_name):
        super().__init__(env_name)

        self.updater = utils.updaters.get_updater(self.conf)

        # get num actions
        env = utils.get_env(env_name)
        num_actions = env.action_space.n
        env.close()

        self.values = utils.data_structures.ValueMap(num_actions, init_val=self.conf['init_val'])
        self.num_actions = num_actions

        self.weight_path = Path("./weights/" + self.conf['save_name'])

        self.discretizer = discretizers.get_discretizer(env_name)

    def act(self, state):

        if np.random.random() < self.conf['epsilon']:
            return np.random.randint(self.num_actions)

        return self.act_optimal(state)

    def act_optimal(self, state):
        state = self.discretizer(state)
        return np.argmax(self.values.get(state))

    def load_if_exists(self):
        if self.weight_path.exists():
            self.load()

    def save(self):
        self.values.save(self.weight_path)

    def load(self):
        self.values.load(self.weight_path)

    def reset(self):
        remove(self.weight_path)

    def learn(self, episode_transitions):
        
        last_transition: utils.types.transition = episode_transitions[-1]

        s = self.discretizer(last_transition.state)
        a = last_transition.action
        r = last_transition.reward

        next_state = self.discretizer(last_transition.next_state)

        done = last_transition.done

        next_val = 0
        if done:
            # zero out the value of final state
            self.values.zero_out(next_state)
        else:
            next_val = np.max(self.values.get(next_state))

        old_val = self.values.get(s)[a]
        n_visits = self.values.get_visits(s)[a]

        updated = self.updater(old_val, r + self.conf['gamma'] * next_val, n_visits)


        self.values.set(s, a, updated)
        self.values.update_count(s, a)





