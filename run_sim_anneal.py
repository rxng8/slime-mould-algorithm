
# %%
import numpy as np

from lib import Config
from lib.algorithms import SimulatedAnnealing as SA
from lib.benchmarks import Rosenbrock
from lib.solve import solve
from lib.utils import generate_latex_table

# Global variables (experiment setup)
SEED = 1990 #123 #1234 #69
D = 2
LB = -2.0
UB = 2.0
TRIALS = 30
MAX_I = 1000

np.random.seed(SEED)

# Configuration values for optimization algorithm
config = Config(
  D = D,
  stop_criterion = {
    "type": "temperature",
    "min_temperature": 0.1
  },
  initializer = "uniform",
  cooling = "geometric",
  T_f = 1e-10,
  t_f = 1000, # For linear cooling (maybe change as optional)
  T_0 = 0,
  lb = LB,
  ub = UB,
  alpha = 0.1,
  a = 0.99,
  funct = Rosenbrock
)

T_0s = [100, 50, 25, 10, 1]
hyperparam_list = [
    {'alpha': 0.1},
    {'alpha': 0.08},
    {'alpha': 0.05},
    {'alpha': 0.01},
]

results = {size: [] for size in T_0s}
headers = ['Init T'] + [f'$F={hp["alpha"]}$' for hp in hyperparam_list]
experiment_name = 'Test_SA'

for i, T_0 in enumerate(T_0s):
  config = config.update(T_0=T_0)
  _, mean_result, std_dev_result, latex_result = solve(TRIALS, SA, config, log_to_file=True, experiment_name=experiment_name)
  print(f"Trial: {i} with T_0={T_0}, mean={mean_result}, std={std_dev_result}")
  results[T_0].append(latex_result)

generate_latex_table(results, headers, experiment_name)