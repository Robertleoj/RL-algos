from dataclasses import dataclass
import numpy as np


@dataclass
class transition:
    state: np.ndarray
    action: any
    reward: float
    next_state: np.ndarray
    done: bool
