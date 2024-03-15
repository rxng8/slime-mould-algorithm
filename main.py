
# %%

from typing import List, Dict, Tuple, Callable, Sequence, Any
import numpy as np

from lib import Config
from lib.types import MetricsType
from lib.algorithms import SlimeMould

config = Config(
  pop_size = 100,
  dim = 2,
  initializer = "normal",
  seed = 1990,
)
config = config.update(pop_size=50)
print(config)


S = SlimeMould(config)



