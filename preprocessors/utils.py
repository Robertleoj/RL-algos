import numpy as np


class Normalizer:
    def __init__(self, lower_bounds: np.ndarray, upper_bounds:np.ndarray):
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.range = upper_bounds - lower_bounds

    def normalize(self, state: np.ndarray):
        # normalize between -1 and 1
        return ((state - self.lower_bounds) / self.range) * 2 - 1



