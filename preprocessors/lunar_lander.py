import numpy as np
from .utils import Normalizer

class LunarLanderPreprocessor:
    def __init__(self):
        self.normalizer = Normalizer(
            np.array(
                [
                    -1.5, -1.5, -5, -5, -np.pi, -5., -0., -0.
                ]
            ),
            np.array(
                [
                    1.5, 1.5, 5., 5., np.pi, 5., 1., 1.
                ]
            )
        )
    
    def __call__(self, state):
        return self.normalizer.normalize(state)