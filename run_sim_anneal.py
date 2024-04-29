
# %%
import numpy as np

from lib import Config
from lib.algorithms import SimulatedAnnealing as SA
from lib.benchmarks import Rosenbrock
from lib.solve import solve

# Global variables (experiment setup)
SEED = 1990 #123 #1234 #69
D = 2
LB = -2.0
UB = 2.0
TRIALS = 30,
MAX_I = 1000,

np.random.seed(SEED)

# Configuration values for optimization algorithm
config = Config(
  dim = D,
  stop_criterion = {
    "type": "temperature",
    "min_temperature": 0.1
  },
  initializer = "uniform",
  cooling = "geometric",
  population=30,
  T_0 = 100,
  T_f = 1e-10,
  t_f = 1000, # For linear cooling (maybe change as optional)
  lb = LB,
  ub = UB,
  alpha = 0.1,
  a = 0.99,
  funct = Rosenbrock
)

print(config)
experiment_name = 'Test_SA'

_, mean_result, std_dev_result, latex_result = solve(30, SA, config, log_to_file=True, experiment_name=experiment_name)
print(mean_result, )