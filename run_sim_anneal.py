
# %% this is from Matt

from typing import List, Dict, Tuple, Callable, Sequence, Any
import numpy as np

from lib import Config
from lib.types import MetricsType
from lib.algorithms import SimulatedAnnealing
from lib.benchmarks import Rosenbrock

config = Config(
  trials = 30,
  dim = 2,
  initializer = "uniform",
  cooling = "geometric",
  seed = 1990,
  T_0 = 100,
  T_f = 1e-10,
  t_f = 1000, # For linear cooling (maybe change as optional)
  lb = -2.0,
  ub = 2.0,
  alpha = 0.1,
  a = 0.99,
  f = Rosenbrock
)

print(config)


S = SimulatedAnnealing(config)
S.run()
