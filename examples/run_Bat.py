import os, sys
import numpy as np
from typing import Dict, List

from lib.config import Config
from lib.algorithms import Bat
from lib.benchmarks.functions import rosenbrock
from lib.solve import solve
from lib.utils import generate_latex_table

# Add the lib directory to the sys.path
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, lib_path)

# Best RESULTS rosenbrock:
# pop_size = 25
# alpha = 1.0
# pulse_rate = 0.2
# gamma = 0.9
# => mean=4.592837426532917e-07, std=4.4653484706698026e-07

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
            {'type': 'fitness', 'target_fitness': 0.01}
        ]
    },
    min="Minimization",
    gamma=0.0,
    alpha=0.0,
    pulse_rate=0.0,
    population=0,
    freq_min=0,
    freq_max=2,
    loudness=0.25,
)

pop_bats = [5, 10, 25, 50, 100]

# different hyper-parameters used
# alpha, gamma, pulse rate
hyperparam_list = [
    {'alpha': 0.9, 'gamma': 0.9, 'pulse_rate':0.5},
    {'alpha': 0.9, 'gamma': 0.7, 'pulse_rate':0.7},
    {'alpha': 1.0, 'gamma': 0.9, 'pulse_rate':0.2}
]

results = {size: [] for size in pop_bats}
# Headers based on hyperparameter configurations
headers = ['Popul. Size'] + [f'$\\alpha={hp["alpha"]}, \\gamma={hp["gamma"]}, \\pulse rate={hp["pulse_rate"]}$' for hp in hyperparam_list]

# Experiment logging info
experiment_name = 'Test_Bats'

for pop_bat in pop_bats:
    config = config.update(population=pop_bat)
    
    for hyper_params in hyperparam_list:
        config = config.update(hyper_params)
        # Run experiment
        _, mean_result, std_dev_result, latex_result = solve(30, Bat, config, log_to_file=True, experiment_name=experiment_name)
        results[pop_bat].append(latex_result)
        print(f"with Population={pop_bat}, Alpha={hyper_params['alpha']}, Pulse Rate={hyper_params['pulse_rate']}, Gamma={hyper_params['gamma']}, mean={mean_result}, std={std_dev_result}")

generate_latex_table(results, headers, experiment_name)