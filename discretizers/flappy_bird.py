import numpy as np
from .utils import DiscreteRange, DiscreteInf

config = {
    'to_pipe': {
        'start': 0,
        'end': 2,
        'num_intervals': 100
    },
    'vert_dist': {
        'start': -0.3,
        'end': 0.7,
        'num_intervals': 100
    }
}

class FlappyBirdDiscretizer:
    def __init__(self):
        self.to_pipe = DiscreteRange.from_config(config['to_pipe'])
        self.vert_dist = DiscreteRange.from_config(config['vert_dist'])
    

    def __call__(self, observation: np.ndarray):
        to_pipe, vert_dist = observation.tolist()

        to_pipe = self.to_pipe(to_pipe)
        vert_dist = self.vert_dist(vert_dist)


        return (to_pipe, vert_dist)
