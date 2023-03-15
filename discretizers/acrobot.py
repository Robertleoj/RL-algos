import numpy as np
from .utils import DiscreteRange


config = {
    'theta1': {
        'start': -np.pi,
        'end': np.pi,
        'num_intervals': 20
    },

    'theta2': {
        'start': -np.pi,
        'end': np.pi,
        # 'num_intervals': 20
        'num_intervals': 30
    },

    'theta1_angular_vel': {
        'start': -12.566370614359172,
        'end': 12.566370614359172,
        'num_intervals': 25
    },

    'theta2_angular_vel': {
        'start': -28.274333882308138,
        'end': 28.274333882308138,
        'num_intervals': 35
    }
}

class AcrobotDiscretizer:
    def __init__(self):
        self.theta1 = DiscreteRange.from_config(config['theta1'])
        self.theta2 = DiscreteRange.from_config(config['theta2'])
        self.theta1_angular_vel = DiscreteRange.from_config(config['theta1_angular_vel'])

        self.theta2_angular_vel = DiscreteRange.from_config(config['theta2_angular_vel'])


    def __call__(self, observation: np.ndarray):
        cos_theta1, sin_theta1, cos_theta2, sin_theta2, theta1_angular_vel, theta2_angular_vel = observation.tolist()

        theta1 = np.arctan2(sin_theta1, cos_theta1)
        theta2 = np.arctan2(sin_theta2, cos_theta2)

        theta1 = self.theta1(theta1)
        theta2 = self.theta2(theta2)

        theta1_angular_vel = self.theta1_angular_vel(theta1_angular_vel)

        theta2_angular_vel = self.theta2_angular_vel(theta2_angular_vel)

        return (theta1, theta2, theta2_angular_vel, theta1_angular_vel)

