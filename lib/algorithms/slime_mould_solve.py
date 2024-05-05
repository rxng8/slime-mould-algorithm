
# external
import numpy as np
from typing import Callable
from lib.types import MetricsType
from lib.config import Config
from lib.utils import define_stop_criterion

class SlimeMould:
  def __init__(self, config: Config) -> None:
    self.config = config
    self.objective_fn = config.funct
    self.stop: Callable = define_stop_criterion(config)


  def __fitness(self, X):
    # This algorithm is defaulted to maximizing the objective function
    minimizing = self.config.get("minimizing", False)
    if minimizing:
      return -self.objective_fn(X) # (B, 1)
    else:
      return self.objective_fn(X)

  def __sort_slime_mould(self, X, F):
    # X: (B, D); F: (B, 1)
    _F = F[:, 0] # (B,)
    sorted_idx = np.argsort(_F, 0)[::-1]
    return X[sorted_idx], F[sorted_idx] # (B, D), (B, 1) decrementally

  def __vb(self, t):
    # https://www.desmos.com/calculator/mgfdu3tpmn
    # NOTE: have to plus one here because the formula used 1-system
    return np.arctanh(-(float(t+1) / self.config.max_iters) + 1)

  def __vc(self, t):
    return 1 - float(t + 1) / self.config.max_iters

  def step(self, t, X, W, DF):
    ### Approach food
    F = self.__fitness(X) # fitness of all slime mould
    X, F = self.__sort_slime_mould(X, F)
    bF = F[0][0] # scalar
    wF = F[-1][0] # scalar

    tricky_term = np.random.uniform() * np.log((bF - F) / (bF - wF).clip(1e-8) + 1)
    # print(f"[DEBUG] bF - F: {(bF - F).mean()}; bF - wF: {(bF - wF).mean()}; all: {((bF - F) / (bF - wF).clip(1e-8) + 1).mean()}")
    mid = self.config.pop_size // 2
    W_new = W
    W_new[:mid] = 1 - tricky_term[:mid]
    W_new[mid:] = 1 + tricky_term[mid:]

    ### Wrap food and Oscilation
    iA, iB = np.random.choice(self.config.pop_size, 2, replace=False)
    XA, XB = X[iA][None], X[iB][None] # (1, D)
    Xb = X[0][None] # (1, D)
    p = np.tanh(np.abs(F - DF)) # (B, 1)
    p = np.repeat(p, self.config.D, axis=-1)
    r = np.random.uniform()
    X_new_1 = Xb + self.__vb(t) * (W * XA - XB)
    X_new_2 = self.__vc(t) * X
    X_new = np.select([r < p, r >= p], [X_new_1, X_new_2])

    DF = bF if bF > DF else DF
    return X_new.clip(self.config.lb, self.config.ub), W_new, DF

  def solve(self) -> tuple:
    X = np.random.normal(0, 1, (self.config.pop_size, self.config.D)).clip(self.config.lb, self.config.ub) # locations/population/slime mould
    W = np.random.normal(0, 1, (self.config.pop_size, 1)) # weights
    DF = np.max(self.__fitness(X), 0)[0] # scalar

    #state = {'current_fitness': -DF}
    for t in range(self.config.max_iters):
      # if self.stop(state):
      #   break
      X, W, DF = self.step(t, X.copy(), W.copy(), DF)
      #state.update({'current_fitness': -DF})
      
    id = np.argmax(self.__fitness(X), 0)[0]
    return X[id], -DF


