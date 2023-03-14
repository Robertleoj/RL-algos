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
        cart_pos_conf = config['cart_pos']
        self.cart_pos = DiscreteRange(
            cart_pos_conf['start'], 
            cart_pos_conf['end'], 
            cart_pos_conf['num_intervals']
        )

        pole_angle_conf = config['pole_angle']
        self.pole_angle = DiscreteRange(
            pole_angle_conf['start'],
            pole_angle_conf['end'],
            pole_angle_conf['num_intervals']
        )

        pole_angular_vel_conf = config['pole_angular_vel']
        self.pole_angular_vel = DiscreteInf(
            pole_angular_vel_conf['num_intervals'],
            pole_angular_vel_conf['factor']
        )

        cart_vel_conf = config['cart_vel']
        self.cart_vel = DiscreteInf(
            cart_vel_conf['num_intervals'],
            cart_vel_conf['factor']
        )
    

    def __call__(self, observation: np.ndarray):
        cart_pos, cart_vel, pole_angle, pole_angular_vel = observation.tolist()


        cart_pos = self.cart_pos(cart_pos)
        pole_angle = self.pole_angle(pole_angle)
        pole_angular_vel = self.pole_angular_vel(pole_angular_vel)
        cart_vel = self.cart_vel(cart_vel)

        return (cart_pos, cart_vel, pole_angle, pole_angular_vel)
