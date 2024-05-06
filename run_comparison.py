# %%

import numpy as np
from typing import Dict, List

from lib.config import Config
from lib.algorithms import FireFly, Bat, PSO, SlimeMould, SimulatedAnnealingSlimeMould
from lib.benchmarks.functions import rosenbrock
import lib.benchmarks.functions as function_module
from inspect import getmembers, isfunction
from lib.solve import compare

# FUNCT = rosenbrock

# Loop through all the benchmark
for (func_name, FUNCT) in getmembers(function_module, isfunction):

    # Global variables (experiment setup)
    SEED = 1990
    TRIALS = 30
    GLOBAL = {
        'D': 2,
        'lb': -2.0,
        'ub': 2.0,
        'MAX_I': 1000,
        'funct': FUNCT,
        'min' :"Minimization",
        'minimizing': FUNCT.minimizing,
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

    slime_mould_config = Config(
        max_iters=1000
    )
    slime_mould_config = slime_mould_config.update(GLOBAL)

    simulated_annealing_slime_mould_config = Config(
        max_iters=1000,
        cooling_rate=0.8
    )
    simulated_annealing_slime_mould_config = simulated_annealing_slime_mould_config.update(GLOBAL)

    configs = [firefly_config, bat_config, pso_config, slime_mould_config, simulated_annealing_slime_mould_config]
    algos = [FireFly, Bat, PSO, SlimeMould, SimulatedAnnealingSlimeMould]

    compare(algos, configs, populations, TRIALS, f"{func_name}")

