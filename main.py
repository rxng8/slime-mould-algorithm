
# %%

from typing import List, Dict, Tuple, Callable, Sequence, Any
import numpy as np

from lib import Config
from lib.types import MetricsType
from lib.benchmarks import Ackley, Rosenbrock, negative_Alpine
from lib.algorithms import SlimeMould

# initial config
config = Config(
  pop_size = 1000,
  dim = 10,
  max_iters = 20,
  minimizing = True,
  seed = 69,
  lower_bound=-5.0,
  upper_bound=5.0
)

np.random.seed(config.seed)

####### Run Ackley #####
print("\nAckley:")
fn = Ackley
config = config.update(minimizing=fn.minimizing)
print(config)
S = SlimeMould(fn, config)
S.run()

####### Run Rosenbrock #####
print("\nRosenbrock:")
fn = Rosenbrock
config = config.update(
  minimizing=fn.minimizing,
  max_iters=100,
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
  minimizing=fn.minimizing,
  max_iters=50,
  dim=2,
  lower_bound=-10.0,
  upper_bound=10.0
)
print(config)
S = SlimeMould(fn, config)
S.run()

