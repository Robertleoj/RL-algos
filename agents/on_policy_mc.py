import numpy as np
from pathlib import Path
import discretizers
from .base import AgentBase
import utils
from os import remove

class OnPolicyMonteCarlo(AgentBase):
    agent_name = 'on_policy_mc'

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

    def learn_episode_end(self, episode_transitions):

        # zero out return of last transition
        last_transition = episode_transitions[-1]
        if last_transition.done:
            self.values.zero_out(self.discretizer(last_transition.next_state))

        ep_return = 0


        counts: dict = {}

        for t in episode_transitions:
            state = self.discretizer(t.state)
            action = t.action

            if (state, action) not in counts:
                counts[(state, action)] = 0

            counts[(state, action)] += 1

        for transition in reversed(episode_transitions):

            dict_el = (self.discretizer(transition.state), transition.action)

            if counts[dict_el] != 1:
                counts[dict_el] -= 1
                continue

            ep_return = transition.reward + self.conf['gamma'] * ep_return

            state = self.discretizer(transition.state)
            action = transition.action


            old_val = self.values.get(state)[action]

            n_visits = self.values.get_visits(state)[action]

            new_val = self.updater(old_val, ep_return, n_visits)

            self.values.set(state, action, new_val)

            self.values.update_count(state, action)

