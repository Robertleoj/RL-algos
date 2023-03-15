import numpy as np
from pathlib import Path
import utils
from ..base import AgentBase
import discretizers
from os import remove


class DynaQ(AgentBase):
    agent_name = 'DynaQ'

    def __init__(self, env_name):
        super().__init__(env_name)

        self.updater = utils.updaters.get_updater(self.conf)
        self.discretizer = discretizers.get_discretizer(env_name)

        # get num actions
        env = utils.get_env(env_name)
        num_actions = env.action_space.n
        env.close()

        self.values = utils.data_structures.ValueMap(num_actions, init_val=self.conf['init_val'])

        self.buffer = utils.data_structures.DynaBuffer(self.conf['buffer_size'])

        self.num_actions = num_actions

        self.weight_path = Path("./weights/" + self.conf['save_name'])
        self.buffer_path = Path("./buffers/" + self.conf['buffer_save_name'])

    def act(self, state: np.ndarray):

        if np.random.random() < self.conf['epsilon']:
            return np.random.randint(self.num_actions)

        return self.act_optimal(state)

    def act_optimal(self, state):
        state = self.discretizer(state)
        return np.argmax(self.values.get(state))

    def load_if_exists(self):
        if (
            self.weight_path.exists() 
            and self.buffer_path.exists()
        ):
            self.load()

    def save(self):
        self.values.save(self.weight_path)
        self.buffer.save(self.buffer_path)

    def load(self):
        self.values.load(self.weight_path)
        self.buffer.load(self.buffer_path)

    def reset(self):
        if self.weight_path.exists():
            remove(self.weight_path)
        if self.buffer_path.exists():
            remove(self.buffer_path)

    def learn(self, episode_transitions):
        last_transition: utils.types.transition = episode_transitions[-1]

        s = self.discretizer(last_transition.state)
        a = last_transition.action
        r = last_transition.reward
        s_next = self.discretizer(last_transition.next_state)
        done = last_transition.done


        self.__learn(s, a, r, s_next, done)

        self.buffer.insert((s, a, r, s_next, done))

        self.plan()
        

    def __learn(self, state, action, reward, next_state, done, update_count = True):


        next_val = 0
        if not done:
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
            s, a, r, s_next, done = sample
            self.__learn(s, a, r, s_next, done, update_count=True)
