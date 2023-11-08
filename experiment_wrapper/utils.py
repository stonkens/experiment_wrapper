from experiment_wrapper import Controller
import numpy as np


class InactiveController(Controller):
    def __init__(self, dims: int):
        self.dims = dims

    def __call__(self, x, t):
        return np.zeros((x.shape[0], self.dims))
