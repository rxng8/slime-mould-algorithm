# %%
import numpy as np

from lib import Config
from lib.algorithms import PSO
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

# Configuration for PSO
config = Config(
    D=D,
    lb=LB,
    ub=UB,
    funct=Rosenbrock,
    stop_criterion={
        'type': 'complex',
        'criteria': [
            {'type': 'iterations', 'max_iterations': 1000},
            {'type': 'fitness', 'target_fitness': 1e-6}  # Assume optima=0
        ]
    },
    v_max=1.0,  # Maximum velocity
    v_init=0.1,  # Initial max velocity
    alpha=0.0,  # Cognitive coefficient
    beta=0.0,  # Social coefficient
    population=0  # Population size
)

pop_sizes = [5, 10, 25, 50, 100]
hyperparam_list = [
    {'alpha': 0.9, 'beta': 0.5},
    {'alpha': 0.4, 'beta': 0.5},
    {'alpha': 0.9, 'beta': 0.1},
    {'alpha': 0.9, 'beta': 0.8},
]

results = {size: [] for size in pop_sizes}
headers = ['Popul. Size'] + [f'$\\alpha={hp["alpha"]}, \\beta={hp["beta"]}$' for hp in hyperparam_list]
# Experiment logging info
experiment_name = 'Test_PSO'

for pop_size in pop_sizes:
    config = config.update(population=pop_size)

    for hyper_params in hyperparam_list:
        config = config.update(hyper_params)
        
        # Run experiment
        _, mean_result, std_dev_result, latex_result = solve(TRIALS, PSO, config, log_to_file=True, experiment_name=experiment_name)
        results[pop_size].append(latex_result)
        print(f"with Population={pop_size}, Alpha={hyper_params['alpha']}, Beta={hyper_params['beta']}, mean={mean_result}, std={std_dev_result}")

generate_latex_table(results, headers, experiment_name)
