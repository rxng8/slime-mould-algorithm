
# %%

from typing import List, Dict, Tuple, Callable, Sequence, Any
import numpy as np

from lib import Config
from lib.types import MetricsType
from lib.benchmarks import Ackley
from lib.algorithms import SlimeMould

config = Config(
  pop_size = 100,
  dim = 10,
  max_iters = 20
)
print(config)
S = SlimeMould(lambda x: -Ackley(x), config)
S.run()


