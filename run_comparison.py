import numpy as np
from typing import Dict, List

from lib.config import Config
from lib.algorithms import FireFly, Bat, PSO
from lib.benchmarks.functions import rosenbrock
from lib.solve import compare

# Global variables (experiment setup)
SEED = 1990
TRIALS = 30
GLOBAL = {
    'D': 2,
    'lb': -2.0,
    'ub': 2.0,
    'MAX_I': 1000,
    'funct': rosenbrock,
    'min' :"Minimization",
    'stop_criterion': {
        'type': 'complex',
        'criteria': [
            {'type': 'iterations', 'max_iterations': 1000},
            {'type': 'fitness', 'target_fitness': 1e-6}
        ]
    }
}

np.random.seed(SEED)

populations = [5, 10, 25, 50, 100]

firefly_config = Config(gamma=1.0, alpha=0.5, beta0=1.0)
firefly_config = firefly_config.update(GLOBAL)

bat_config = Config(
    alpha=1.0, 
    pulse_rate=0.2, 
    gamma=0.9,
    freq_min=0,
    freq_max=2,
    loudness=0.25,
    )
bat_config = bat_config.update(GLOBAL)

pso_config = Config(
    alpha=0.9, 
    beta=0.5,
    v_max=1.0,  # Maximum velocity
    v_init=0.1,  # Initial max velocity
    )
pso_config = pso_config.update(GLOBAL)

configs = [firefly_config, bat_config, pso_config]
algos = [FireFly, Bat, PSO]

compare(algos, configs, populations, TRIALS, "Test_Compare")