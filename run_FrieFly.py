import numpy as np
from typing import Dict, List

from lib.config import Config
from lib.algorithms import FireFly
from lib.benchmarks import Rosenbrock
from lib.solve import solve
from lib.utils import generate_latex_table

# Global variables (experiment setup)
SEED = 1990
D = 2
LB = -2.0
UB = 2.0
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
            {'type': 'iterations', 'max_iterations': 1000},
            {'type': 'fitness', 'target_fitness': 0.01}
        ]
    },
    min="Minimization",
    gamma=0.0,
    alpha=0.0,
    beta0=0.0,
    population=0,
)

swarm_sizes = [5, 10, 25, 50, 100]
# different hyper-parameters used
hyperparam_list = [
    {'gamma': 1.0, 'alpha': 0.5, 'beta0':1.0},
    {'gamma': 1.0, 'alpha': 0.2, 'beta0':1.0},
    {'gamma': 1.0, 'alpha': 0.2, 'beta0':0.2}
]

results = {size: [] for size in swarm_sizes}
# Headers based on hyperparameter configurations
headers = ['Popul. Size'] + [f'$\\gamma={hp["gamma"]}, \\alpha={hp["alpha"]}, \\beta_0={hp["beta0"]}$' for hp in hyperparam_list]

# Experiment logging info
experiment_name = 'test_FireFly'

for swarm_size in swarm_sizes:
    config = config.update(population=swarm_size)
    
    for hyper_params in hyperparam_list:
        config = config.update(hyper_params)
        # Run experiment
        _, mean_result, std_dev_result, latex_result = solve(TRIALS, FireFly, config, log_to_file=True, experiment_name=experiment_name)
        results[swarm_size].append(latex_result)
        
generate_latex_table(results, headers, experiment_name)