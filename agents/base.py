from config import config
import signal


class AgentBase:
    agent_name = None
    def __init__(self, env_name):
        self.env_name = env_name
        self.conf = config[self.agent_name][env_name]

    def play(self):
        self.load()
        raise NotImplementedError

    def train(self, load=False):
        self.save_on_exit()
        raise NotImplementedError

    def save(self):
        pass

    def load(self):
        pass

    def save_on_exit(self):

        def signal_handler(sig, frame):
            self.save()
            exit(0)

        signal.signal(signal.SIGINT, signal_handler)