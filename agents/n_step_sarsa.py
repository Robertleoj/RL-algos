import numpy as np
from pathlib import Path
import discretizers
from .base import AgentBase
import utils
from os import remove

class NStepSarsa(AgentBase):
    agent_name = 'n_step_sarsa'

    def __init__(self, env_name:str):
        super().__init__(env_name)

        self.discretizer = discretizers.get_discretizer(env_name)
        self.updater = utils.updaters.get_updater(self.conf)

        # get num actions
        env = utils.get_env(env_name)
        num_actions = env.action_space.n
        env.close()

        self.values = utils.data_structures.ValueMap(num_actions, init_val=self.conf['init_val'])

        self.num_actions = num_actions

        self.weight_path = Path("./weights/" + self.conf['save_name'])

    def episode_print(self):
        print("num state-action pairs: ", self.values.num_pairs())

    
    def act(self, state: np.ndarray):
        if np.random.random() < self.conf['epsilon']:
            return np.random.randint(self.num_actions)
        
        return self.act_optimal(state)

    def act_optimal(self, state: np.ndarray):
        state = self.discretizer(state)
        return np.argmax(self.values.get(state))

    def load_if_exists(self):
        if self.weight_path.exists():
            self.load()

    def load(self):
        self.values.load(self.weight_path)

    def save(self):
        self.values.save(self.weight_path)

    def reset(self):
        if self.weight_path.exists():
            remove(self.weight_path)

    def __learn_bootstrap(self, transitions, with_final_state=False):

        g = 0

        loop_transitions = transitions
        if not with_final_state:
            loop_transitions = transitions[:-1]
        

        for i, transition in enumerate(loop_transitions):
            g += transition.reward * (self.conf['gamma'] ** i)

        if with_final_state:
            g += transitions[-1].reward * (self.conf['gamma'] ** len(transitions))
        else:
            s = self.discretizer(transitions[-1].state)
            a = transitions[-1].action

            g += self.values.get(s)[a] * (self.conf['gamma'] ** (len(transitions) - 1))

        s = self.discretizer(transitions[0].state)
        a = transitions[0].action

        old_q = self.values.get(s)[a]

        num_encounters = self.values.get_visits(s)[a]

        new_q = self.updater(old_q, g, num_encounters)

        self.values.set(s, a, new_q)
        self.values.update_count(s, a)

    def learn_episode_end(self, episode_transitions):
        num_trans = len(episode_transitions)

        for i in range(min(self.conf['n'], num_trans) + 1):
            learn = episode_transitions[-i:]
            self.__learn_bootstrap(learn, True)


    def learn(self, episode_transitions: list):
        if len(episode_transitions) < self.conf['n']:
            return

        if episode_transitions[-1].done:
            return

        learn = episode_transitions[-self.conf['n'] - 2:]

        self.__learn_bootstrap(learn, False)

        

