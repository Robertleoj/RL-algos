from .acrobot import AcrobotDiscretizer
from .cartpole import CartPoleDiscretizer
from .mountain_car import MountainCarDiscretizer
from .flappy_bird import FlappyBirdDiscretizer
from .taxi import TaxiDiscretizer

def get_discretizer(env_name):
    match env_name:
        case 'CartPole-v1':
            return CartPoleDiscretizer()
        case 'MountainCar-v0':
            return MountainCarDiscretizer()
        case 'Acrobot-v1':
            return AcrobotDiscretizer()
        case 'Taxi-v3':
            return TaxiDiscretizer()
        case 'FlappyBird-v0':
            return FlappyBirdDiscretizer()
        case _:
            raise ValueError(f"Unrecognized environment: {env_name}")


