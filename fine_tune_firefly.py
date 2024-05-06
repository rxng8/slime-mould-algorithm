import numpy as np
from typing import Dict, List

from lib.config import Config
from lib.algorithms import FireFly
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

hyperparam_list = [
    {'gamma': 1.0, 'alpha': 0.5, 'beta0':1.0},
    {'gamma': 1.0, 'alpha': 0.2, 'beta0':1.0},
    {'gamma': 1.0, 'alpha': 0.2, 'beta0':0.2}
]

# Experiment logging info
experiment_name = 'test_FireFly'

config = Config(
    D=2,
    stop_criterion = {
        'type': 'complex',
        'criteria': [
            {'type': 'iterations', 'max_iterations': 1000},
            {'type': 'fitness', 'target_fitness': -1.0} # depends on function!
        ]
    }
)
config = config.update(cube)

results = {size: [] for size in populations}
# Headers based on hyperparameter configurations
headers = ['Popul. Size'] + [f'$\\gamma={hp["gamma"]}, \\alpha={hp["alpha"]}, \\beta_0={hp["beta0"]}$' for hp in hyperparam_list]

for population in populations:
    config = config.update(population=population)
    
    for hyper_params in hyperparam_list:
        config = config.update(hyper_params)
        # Run experiment
        _, mean_result, std_dev_result, latex_result = solve(TRIALS, FireFly, config, log_to_file=False, experiment_name=experiment_name)
        results[population].append(latex_result)

generate_latex_table(results, headers, experiment_name)