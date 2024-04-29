# %%

from typing import List, Dict, Tuple, Callable, Sequence, Any
import numpy as np

from lib.types import MetricsType
from lib.utils import get_initializer, define_stop_criterion

class SimulatedAnnealing:
    def __init__(self, config) -> None:
        self.config = config
        self.initializer: Callable = get_initializer(config.initializer)
        self.X_start = self.initializer(config.population, config.dim, config.lb, config.ub)
        self.cooling: Callable = self.init_cooling_schedule(config.cooling)
        self.T = config.T_0
        self.stop = define_stop_criterion(config)
        self.f = config.funct
    
    def init_cooling_schedule(self, name : str) -> Callable:
        return {
        "geometric": lambda t: self.config.T_0 * (self.config.a ** t),
        "linear": lambda t:  self.config.T_0 - t * ((self.config.T_0 - self.config.T_f) / self.config.t_f),
    }[name]
    
    def solve(self):
        X = self.X_start[0] 
        t = 0
        best_X, best_fitness = X, self.f(X)
        
        state = {
            'temperature': self.T,
            'iterations': t,
            'current_fitness': best_fitness
        }
        
        # add this as field to config file        
        while not self.stop(state):
            epsilon = np.random.normal(0, self.config.alpha, size=X.shape)
            X_new = X + epsilon
            X_new = np.clip(X_new, self.config.lb, self.config.ub)  # Ensure X_new stays within bounds
            delta_f = self.f(X_new) - self.f(X)
            
            if delta_f < 0 or (np.exp(- delta_f / self.T) > np.random.uniform(0, 1)):
                X = X_new
                current_fitness = self.f(X)
                if current_fitness < best_fitness:
                    best_X, best_fitness = X, current_fitness
                    
            t += 1
            self.T = self.cooling(t)
            # Update state after changes
            state.update({'temperature': self.T, 'iterations': t})
        return best_X, best_fitness