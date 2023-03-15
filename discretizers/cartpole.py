import numpy as np
from .utils import DiscreteRange, DiscreteInf

config = {
    'cart_pos': {
        'start': -4.8,
        'end': 4.8,
        'num_intervals': 10
    },
    'pole_angle': {
        'start': -0.418,
        'end': 0.418,
        'num_intervals': 20
    },
    'pole_angular_vel': {
        'num_intervals': 20,
        'factor': 9
    },
    'cart_vel': {
        'num_intervals': 20,
        'factor': 9
    }
}

class CartPoleDiscretizer:
    def __init__(self):
        self.cart_pos = DiscreteRange.from_config(config['cart_pos'])

        self.pole_angle = DiscreteRange.from_config(config['pole_angle'])

        self.pole_angular_vel = DiscreteInf.from_config(config['pole_angular_vel'])

        self.cart_vel = DiscreteInf.from_config(config['cart_vel'])
    

    def __call__(self, observation: np.ndarray):
        cart_pos, cart_vel, pole_angle, pole_angular_vel = observation.tolist()


        cart_pos = self.cart_pos(cart_pos)
        pole_angle = self.pole_angle(pole_angle)
        pole_angular_vel = self.pole_angular_vel(pole_angular_vel)
        cart_vel = self.cart_vel(cart_vel)

        return (cart_pos, cart_vel, pole_angle, pole_angular_vel)
