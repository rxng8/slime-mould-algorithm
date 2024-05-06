import os, sys
import numpy as np
from typing import Dict, List

# Add the lib directory to the sys.path
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, lib_path)

from lib.config import Config
from lib.algorithms import FireFly
from lib.benchmarks.functions import rosenbrock
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
    funct=rosenbrock,
    stop_criterion={
        'type': 'complex',
        'criteria': [
            {'type': 'iterations', 'max_iterations': 1000},
            {'type': 'fitness', 'target_fitness': 1e-6}
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
        print(f"with Population={swarm_size}, Alpha={hyper_params['alpha']}, Beta0={hyper_params['beta0']}, Gamma={hyper_params['gamma']}, mean={mean_result}, std={std_dev_result}")

        
generate_latex_table(results, headers, experiment_name)