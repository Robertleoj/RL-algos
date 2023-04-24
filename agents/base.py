from config import config
import signal


class AgentBase:
    agent_name = None
    def __init__(self, env_name):
        self.env_name = env_name
        self.conf = config[self.agent_name][env_name]

    def play(self, load=True):
        if load:
            self.load()

    def train(self, load=True):
        if load:
            self.load()
        self.save_on_exit()

    def save(self):
        pass

    def load(self):
        pass

    def save_on_exit(self):

        def signal_handler(sig, frame):
            self.save()
            exit(0)

        signal.signal(signal.SIGINT, signal_handler)