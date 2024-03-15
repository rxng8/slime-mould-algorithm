# %%

from typing import List, Dict, Tuple, Callable, Sequence, Any
import numpy as np

from lib.types import MetricsType
from lib.utils import get_initializer

class SimulatedAnnealing:
    def __init__(self, config) -> None:
        self.config = config
        self.initializer: Callable = get_initializer(config.initializer)
        self.X_start = self.initializer(config.trials, config.dim, config.lb, config.ub) # generate set of starting X
        # cooling here or in iteration?
        self.cooling: Callable = self.init_cooling_schedule(config.cooling)
        self.T = config.T_0
    
    def init_cooling_schedule(self, name : str) -> Callable:
        return {
        "geometric": lambda t: self.config.T_0 * (self.config.a ** t),
        "linear": lambda t:  self.config.T_0 - t * ((self.config.T_0 - self.config.T_f) / self.config.t_f),
    }[name]
    
    def solve(self, X:np.ndarray):
        t = 0
        while self.T > self.config.T_f:
            epsilon = np.random.normal(0, self.config.alpha, size=X.shape)

            X_new = X + epsilon
            delta_f = self.config.f(X_new) - self.config.f(X)
            
            if delta_f < 0 or (np.exp(- delta_f / self.T) > np.random.uniform(0, 1)):
                X = X_new

            t += 1
            self.T = self.cooling(t)
        return X
                
    
    def run(self):
        for X in self.X_start:
            result = self.solve(X)
            print("X =", result, "f(X) =", self.config.f(result))