# %%

from typing import List, Dict, Tuple, Callable, Sequence, Any
import numpy as np

from lib.types import MetricsType
from lib.utils import get_initializer

def cooling_schedule(name : str) -> Callable:
    return {
        "geometric": lambda T_0, a, t: T_0 * (a ** t),
        "linear": lambda T_0, T_f, t_f, t:  T_0 - t * ((T_0 - T_f) / t_f),
    }[name]

class SimulatedAnnealing:
    def __init__(self, config) -> None:
        # config here needs to hold:
        # initial temps (array of T_0s)
        # number of trials
        # stepsize alpha
        # cooling schedule
        self.config = config
        initializer: Callable = get_initializer(config.initializer)
        self.X_start = initializer(config.trials, config.dim) # generate set of starting X
        
        # cooling here or in iteration?
        cooling: Callable = cooling_schedule(config.cooling)
        self.T = None # Initialize in iterator with init temp from config
    
    def step(self):
        # Here handle the cooling given cooling schedule
        pass
    
    def run(self):
        pass