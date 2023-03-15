import numpy as np
from pathlib import Path
import discretizers
from .base import AgentBase
import utils
from os import remove


class OffPolicyMonteCarlo(AgentBase):
    agent_name = 'off_policy_mc'

    def __init__(self, env_name:str):
        super().__init__(env_name)

        self.discretizer = discretizers.get_discretizer(env_name)

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
        rho = 1

        for transition in reversed(episode_transitions):

            s = self.discretizer(transition.state)
            a = transition.action
            r = transition.reward


            ep_return = r + self.conf['gamma'] * ep_return

            self.values.update_count(s, a, rho)

            lr = rho / self.values.get_visits(s)[a]
            old_q = self.values.get(s)[a]
            new_val = old_q + lr * (ep_return - old_q)

            self.values.set(s, a, new_val)

            optimal_action = np.argmax(self.values.get(s))

            if a == optimal_action:
                rho *= (1 - self.conf['epsilon'] * self.num_actions)
            else:
                break
            