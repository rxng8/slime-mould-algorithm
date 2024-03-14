
# %%

from typing import List, Dict, Tuple, Callable, Sequence, Any
import numpy as np

from lib.types import MetricsType

def get_initializer(name: str) -> Callable:
  return {
    "normal": lambda pop_size, dim: np.random.normal((pop_size, dim)),
    "zeros": lambda pop_size, dim: np.zeros((pop_size, dim)),
  }[name]

class SlimeMould:
  def __init__(self, config) -> None:
    self.config = config
    initializer: Callable = get_initializer(config.initializer) # E.g., config.initializer = "zeros"
    self.X = initializer(config.pop_size, config.dim)
    w_initializer: Callable = None # TODO: put something here
    self.W = None # TODO: put something here
    self.metrics: MetricsType = {}

  def __approach_food(self,):
    pass

  def __wrap_food(self,):
    pass

  def __oscillation(self,):
    pass

  def step(self,):
    max_iterations = self.config.max_iterations
    # Perform one MDP step/iteration
    # modify self.X in one iteration
    pass

  def run(self) -> MetricsType:
    # Run the whole slime mould algorithm using multiple steps.
    pass



