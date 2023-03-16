import numpy as np
from pathlib import Path
import utils
from ..base import AgentBase
import discretizers
from os import remove
from enum import Enum



class DynaQPrioritized(AgentBase):
    agent_name = 'DynaQ_prioritized'

    def __init__(self, env_name):
        super().__init__(env_name)

        self.updater = utils.updaters.get_updater(self.conf)
        self.discretizer = discretizers.get_discretizer(env_name)

        # get num actions
        env = utils.get_env(env_name)
        num_actions = env.action_space.n
        env.close()
        self.num_actions = num_actions

        self.values = utils.data_structures.ValueMap(num_actions, init_val=self.conf['init_val'])

        self.pqbuffer = utils.data_structures.PriorityBuffer()

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
        self.pqbuffer.save(self.buffer_path)

    def load(self):
        self.values.load(self.weight_path)
        self.pqbuffer.load(self.buffer_path)

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

        self.pqbuffer.insert_model(s, a, r, s_next, done)

        diff = self.__learn(s, a, r, s_next, done)
        if diff > self.conf['pq_threshold']:
            self.pqbuffer.insert_pq((s, a), diff)

        self.plan()

    # def episode_print(self):
        # print(f"PQ size: {self.pqbuffer.pq_size()}")
        

    def __learn(self, state, action, reward, next_state, done, update_count = True):


        next_val = 0
        if not done:
            next_val = np.max(self.values.get(next_state))

        old_val = self.values.get(state)[action]
        n_visits = self.values.get_visits(state)[action]

        target = reward + self.conf['gamma'] * next_val

        updated = self.updater(old_val, target, n_visits)

        diff = abs(target - old_val)


        self.values.set(state, action, updated)

        if update_count:
            self.values.update_count(state, action)

        return diff

    def plan(self):
        n = self.conf['n_planning_steps']
        steps_performed = 0
        for _ in range(n):
            if self.pqbuffer.pq_empty():
                break

            s, a = self.pqbuffer.pop_pq()

            r, s_next, done = self.pqbuffer.sample_model(s, a)


            self.__learn(s, a, r, s_next, done, update_count=False)

            for _ in range(self.conf['reverse_samples']):
                s_bar, a_bar = self.pqbuffer.reverse_sample(s)

                r_bar, s_next_bar, done_bar = self.pqbuffer.sample_model(s_bar, a_bar)

                target = r_bar
                if not done_bar:
                    target += self.conf['gamma'] * np.max(self.values.get(s_next_bar))


                curr = self.values.get(s_bar)[a_bar]

                diff = abs(target - curr)

                if diff > self.conf['pq_threshold']:
                    self.pqbuffer.insert_pq((s_bar, a_bar), diff)


            steps_performed += 1

        # print(f"Performed {steps_performed} planning steps")


