# %%
import os, sys
import numpy as np

from lib import Config
from lib.algorithms import DE
from lib.benchmarks import Rosenbrock
from lib.solve import solve
from lib.utils import generate_latex_table

# Add the lib directory to the sys.path
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, lib_path)

# Global variables (experiment setup)
SEED = 1990 #123 #1234 #69
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
            {'type': 'fitness', 'target_fitness': 1e-6} # Assume optima=0
        ]
    },
    crossover_scheme="binomial",
    population=0,
    differential_weight=0.0,
    crossover_param=0.0,
)

pop_sizes = [5, 10, 25, 50, 100]
hyperparam_list = [
    {'differential_weight': 0.9, 'crossover_param': 0.5},
    {'differential_weight': 0.4, 'crossover_param': 0.5},
    {'differential_weight': 0.9, 'crossover_param': 0.1},
    {'differential_weight': 0.9, 'crossover_param': 0.8},
]

results = {size: [] for size in pop_sizes}
headers = ['Popul. Size'] + [f'$F={hp["differential_weight"]}, C_r={hp["crossover_param"]}$' for hp in hyperparam_list]
# Experiment logging info
experiment_name = 'Test_DE'

for pop_size in pop_sizes:
    config = config.update(population=pop_size)

    for hyper_params in hyperparam_list:
        config = config.update(hyper_params)
        # Run experiment
        _, mean_result, std_dev_result, latex_result = solve(30, DE, config, log_to_file=True, experiment_name=experiment_name)
        results[pop_size].append(latex_result)
        print(f"with Population={pop_size}, differential_weight={hyper_params["differential_weight"]}, crossover_param={hyper_params["crossover_param"]}, mean={mean_result}, std={std_dev_result}")

generate_latex_table(results, headers, experiment_name)