import numpy as np
from typing import Dict, List

from lib.config import Config
from lib.algorithms import FireFly, Bat, PSO
import lib.benchmarks.functions as fn
from lib.solve import solve
from lib.utils import generate_latex_table

SEED = 1990
TRIALS = 30

# 1. EXPERIMENT on: negative Alpine
# best tested: gamma=1.0, alpha=0.2, beta_0=1.0
neg_alpine = {
    'funct': fn.negative_Alpine,
    'lb': -10,
    'ub': 10,
    'optima': 0.0,
}

# 2. EXPERIMENT on: ROSENBROCK
# check results
rosenbrock = {
    'funct': fn.rosenbrock,
    'lb': -2.0,
    'ub': 2.0,
    'min' :"Minimization",
    'optima': 0.0,
}

# 3. EXPERIMENT on: ACKLEY
ackley = {
    'funct': fn.ackley,
    'lb': -35.0,
    'ub': 35.0,
    'min': "Minimization",
    'optima': 0.0,
}

# 4. EXPERIMENT on: EASOM (PROBLEM here!)
easom = {
    'funct': fn.easom,
    'lb': -100.0,
    'ub': 100.0,
    'min': "Minimization",
    'optima': -1.0,
}

# 5. EXPERIMENT on: FOUR PEAK
fourpeak = {
    'funct': fn.fourpeak,
    'lb': -5.0,
    'ub': 5.0,
    'optima': 2.0,
}

# 6. EXPERIMENT on: EGGCRATE
eggcrate = {
    'funct': fn.eggcrate,
    'lb': -2.0 * np.pi,
    'ub': 2.0 * np.pi,
    'optima': 0.0,
}

# 7. EXPERIMENT on: BOHACHEVSKY
bohachevsky = {
    'funct': fn.bohachevsky,
    'lb': -100.0,
    'ub': 100.0,
    'min': "Minimization",
    'optima': 0.0,
}

# 8. EXPERIMENT on: BIRD
bird = {
    'funct': fn.bird,
    'lb': -2.0 * np.pi,
    'ub': 2.0 * np.pi,
    'min': "Minimization",
    'optima': -106.764537,
}

# 9. EXPERIMENT on: BARTELSCONN
bartelsconn = {
    'funct': fn.bartelsconn,
    'lb': -500.0,
    'ub': 500.0,
    'min': "Minimization",
    'optima': 1.0,
}

# 10. EXPERIMENT on: BOOTH
booth = {
    'funct': fn.booth,
    'lb': -10.0,
    'ub': 10.0,
    'min': "Minimization",
    'optima': 0.0,
}

# 11. EXPERIMENT on: BRENT
brent = {
    'funct': fn.brent,
    'lb': -10.0,
    'ub': 10.0,
    'min': "Minimization",
    'optima': 0.0,
}

# 12. EXPERIMENT on: BEALE
beale = {
    'funct': fn.beale,
    'lb': -4.5,
    'ub': 4.5,
    'min': "Minimization",
    'optima': 0.0,
}

# 13. EXPERIMENT on: CAMEL
camel = {
    'funct': fn.camel,
    'lb': -5.0,
    'ub': 5.0,
    'min': "Minimization",
    'optima': 0.0,
}

# 14. EXPERIMENT on: BUKIN ??What bounds??
bukin = {
    'funct': fn.bukin,
    'lb': -15.0,
    'ub': -5.0,
    'optima': 0.0,
}

# 15. EXPERIMENT on: CUBE
cube = {
    'funct': fn.cube,
    'lb': -10.0,
    'ub': 10.0,
    'min': "Minimization",
    'optima': 0.0,
}

### Test parameters
populations = [5, 10, 25, 50, 100]

# Parameters for firefly
# hyperparam_list = [
#     # take middle
#     {'gamma': 1.0, 'alpha': 0.5, 'beta0':1.0},
#     {'gamma': 1.0, 'alpha': 0.2, 'beta0':1.0},
#     {'gamma': 1.0, 'alpha': 0.2, 'beta0':0.2}
# ]
# Parameters for bat
# hyperparam_list = [
#     # take middle
#     {'alpha': 0.9, 'gamma': 0.9, 'pulse_rate':0.5, 'freq_min': 0, 'freq_max': 2, 'loudness':0.25},
#     {'alpha': 0.9, 'gamma': 0.7, 'pulse_rate':0.7, 'freq_min': 0, 'freq_max': 2, 'loudness':0.25},
#     {'alpha': 1.0, 'gamma': 0.9, 'pulse_rate':0.2, 'freq_min': 0, 'freq_max': 2, 'loudness':0.25}
# ]
# Parameters for PSO
hyperparam_list = [
    {'alpha': 0.9, 'beta': 0.5, 'v_max':1.0, 'v_init': 0.1},
    {'alpha': 0.4, 'beta': 0.5, 'v_max':1.0, 'v_init': 0.1},
    {'alpha': 0.9, 'beta': 0.1, 'v_max':1.0, 'v_init': 0.1},
    {'alpha': 0.9, 'beta': 0.8, 'v_max':1.0, 'v_init': 0.1},
]

# Experiment logging info
experiment_name = 'test_PSO'

config = Config(
    D=2,
    stop_criterion = {
        'type': 'complex',
        'criteria': [
            {'type': 'iterations', 'max_iterations': 1000},
            {'type': 'fitness', 'target_fitness': -1e-6} # Change depedning on function
        ]
    }
)
config = config.update(neg_alpine) # change function here

results = {size: [] for size in populations}
# Headers for firefly
# headers = ['Popul. Size'] + [f'$\\gamma={hp["gamma"]}, \\alpha={hp["alpha"]}, \\beta_0={hp["beta0"]}$' for hp in hyperparam_list]

# header for bat
# headers = ['Popul. Size'] + [f'$\\alpha={hp["alpha"]}, \\gamma={hp["gamma"]}, \\pulse rate={hp["pulse_rate"]}$' for hp in hyperparam_list]

# header for PSO
headers = ['Popul. Size'] + [f'$\\alpha={hp["alpha"]}, \\beta={hp["beta"]}$' for hp in hyperparam_list]

for population in populations:
    config = config.update(population=population)
    
    for hyper_params in hyperparam_list:
        config = config.update(hyper_params)
        # Run experiment ( with selected algorithm)
        _, mean_result, std_dev_result, latex_result = solve(TRIALS, PSO, config, log_to_file=False, experiment_name=experiment_name)
        results[population].append(latex_result)

generate_latex_table(results, headers, experiment_name)