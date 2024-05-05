
# %%

from typing import List, Dict, Tuple, Callable, Sequence, Any
import numpy as np

from lib import Config
from lib.types import MetricsType
from lib.benchmarks.functions import ackley, rosenbrock, negative_Alpine
from lib.algorithms import SlimeMould

# initial config
config = Config(
  pop_size = 1000,
  dim = 2,
  max_iters = 100,
  minimizing = True,
  seed = 69,
  lower_bound=-5.0,
  upper_bound=5.0
)

np.random.seed(config.seed)

####### Run Ackley #####
print("\nAckley:")
fn = ackley
config = config.update(minimizing=True)
print(config)
S = SlimeMould(fn, config)
S.run()

####### Run Rosenbrock #####
print("\nRosenbrock:")
fn = rosenbrock
config = config.update(
  minimizing=True,
  max_iters=200,
  dim=2,
  lower_bound=-5.0,
  upper_bound=5.0
)
print(config)
S = SlimeMould(fn, config)
S.run()

####### Run Rosenbrock #####
print("\nnegative_Alpine")
fn = negative_Alpine
config = config.update(
  minimizing=False,
  max_iters=50,
  dim=2,
  lower_bound=-10.0,
  upper_bound=10.0
)
print(config)
S = SlimeMould(fn, config)
S.run()

