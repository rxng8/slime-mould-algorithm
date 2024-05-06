import os, sys
import numpy as np
from typing import Dict, List

from lib.config import Config
from lib.algorithms.slime_mould_solve2 import SlimeMould # different slime mould 
from lib.benchmarks.functions import rosenbrock
from lib.solve import solve
from lib.utils import generate_latex_table

# Global variables (experiment setup)
SEED = 69
D = 2
LB = -5.0
UB = 5.0
TRIALS = 30
MAX_I = 200

np.random.seed(SEED)

config = Config(
    D=D,
    lb=LB,
    ub=UB,
    funct=rosenbrock,
    stop_criterion={
        'type': 'fitness', 'target_fitness': 1e-5
        },
    minimizing=True,
    max_iters=MAX_I,
    pop_size=0,  # will update based on experiment
    cooling_rate=0.8
)

#pop_sizes = [1000, 10, 25, 50, 100]
pop_sizes = [1000]
results = {size: [] for size in pop_sizes}
headers = ['Popul. Size'] + [f'$AVG VAL HERE$']

# Experiment logging info
experiment_name = 'test_SlimeMould'

for pop_size in pop_sizes:
    config = config.update(pop_size=pop_size)
    _, mean_result, std_dev_result, latex_result = solve(TRIALS, SlimeMould, config, log_to_file=True, experiment_name=experiment_name)
    results[pop_size].append(latex_result)    
    
generate_latex_table(results, headers, experiment_name)
