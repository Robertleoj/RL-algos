from abc import ABC, abstractmethod
from config import config
import signal
import time
import utils
import numpy as np



class AgentBase:

    agent_name = None
    def __init__(self, env_name):
        self.env_name = env_name
        self.conf = config[self.agent_name][env_name]
        

    @abstractmethod
    def save(self):
        raise NotImplementedError

    @abstractmethod
    def load(self):
        raise NotImplementedError

    @abstractmethod
    def load_if_exists(self):
        raise NotImplementedError

    @abstractmethod
    def learn(self, episode_transitions):
        pass

    @abstractmethod
    def learn_episode_end(self, episode_transitions):
        pass


    @abstractmethod
    def act(self, state: np.ndarray):
        raise NotImplementedError

    @abstractmethod
    def act_optimal(self, state: np.ndarray):
        raise NotImplementedError

    def save_on_exit(self, env):

        def signal_handler(sig, frame):
            self.save()
            env.close()
            exit(0)

        signal.signal(signal.SIGINT, signal_handler)

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    def play(self, games: int=1, load=True):
        env = utils.get_env(self.env_name, train=False)

        if load:
            self.load_if_exists()

        s, _ = env.reset()

        episodes = 0
        while episodes < games:
            env.render()

            if self.env_name == 'FlappyBird-v0':
                time.sleep(1 / 30)
            
            a = self.act_optimal(s)

            s, r, done, _, _ = env.step(a)
            # print(s)
            # print(r)

            if done:
                episodes += 1
                s, _ = env.reset()

        env.close()

    def episode_print(self):
        pass


    def train(self, load=True):
        env = utils.get_env(self.env_name, train=True)

        if load:
            self.load_if_exists()

        s, _ = env.reset()

        self.save_on_exit(env)

        rewards = []

        reward = 0
        episodes = 0
        episode_transitions: list[utils.transition] = []

        for i in utils.N():
            
            a = self.act(s)

            old_s = s
            s, r, done, truncated, info = env.step(a)

            episode_transitions.append(utils.transition(old_s, a, r, s, done))

            reward += r

            self.learn(episode_transitions)

            if i % 10000 == 0:
                print(f"Episode {episodes}")
                if len(rewards) > 0:
                    print(f"Average reward: {sum(rewards[-1000:])/len(rewards[-1000:])}")
                    self.episode_print()

            if done or truncated:
                s, _ = env.reset()
                rewards.append(reward)
                self.learn_episode_end(episode_transitions)
                episode_transitions = []
                reward = 0
                episodes += 1
        

