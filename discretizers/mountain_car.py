import numpy as np
from .utils import DiscreteRange, DiscreteInf

config = {
    'car_pos': {
        'start': -1.2,
        'end': 0.6,
        'num_intervals': 100
    },
    'car_vel': {
        'start': -0.07,
        'end': 0.07,
        'num_intervals': 50
    }
}

class MountainCarDiscretizer:
    def __init__(self):
        self.car_pos = DiscreteRange.from_config(config['car_pos'])
        self.car_vel = DiscreteRange.from_config(config['car_vel'])



    def __call__(self, observation: np.ndarray):
        car_pos, car_vel = observation.tolist()


        car_pos = self.car_pos(car_pos)
        car_vel = self.car_vel(car_vel)

        return (car_pos, car_vel)
