import numpy as np
from typing import Dict, List

from lib.config import Config
from lib.algorithms import SlimeMould
from lib.benchmarks import Rosenbrock
from lib.solve import solve
from lib.utils import generate_latex_table

# Global variables (experiment setup)
SEED = 1990
D = 2
LB = -10.0
UB = 10.0
TRIALS = 30
MAX_I = 1000

np.random.seed(SEED)

config = Config(
    D=D,
    lb=LB,
    ub=UB,
    funct=Rosenbrock,
    stop_criterion={
        'type': 'complex',
        'criteria': [
            {'type': 'iterations', 'max_iterations': MAX_I},
            {'type': 'fitness', 'target_fitness': 0.001}
        ]
    },
    minimizing=Rosenbrock.minimizing,
    max_iters=MAX_I,
    pop_size=0,  # will update based on experiment
)

pop_sizes = [5, 10, 25, 50, 100]
results = {size: [] for size in pop_sizes}
headers = ['Popul. Size'] + [f'$AVG VAL HERE$']

# Experiment logging info
experiment_name = 'test_SlimeMould'

for pop_size in pop_sizes:
    config = config.update(pop_size=pop_size)
    _, mean_result, std_dev_result, latex_result = solve(TRIALS, SlimeMould, config, log_to_file=True, experiment_name=experiment_name)
    results[pop_size].append(latex_result)    
    
generate_latex_table(results, headers, experiment_name)
