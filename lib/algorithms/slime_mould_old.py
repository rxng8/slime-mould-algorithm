

from typing import List, Dict, Tuple, Callable, Sequence, Any
import numpy as np

from lib.types import MetricsType
from lib.config import Config

class SlimeMould:
  def __init__(self, objective_fn: Callable, config: Config) -> None:
    self.config = config
    self.objective_fn = objective_fn

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
    X, F = self.__sort_slime_mould(X, F) # SmellIndex
    bF = F[0][0] # scalar (best fitness)
    wF = F[-1][0] # scalar (worst fitness)

    # Weight update W(SmellIndex)
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
    p = np.repeat(p, self.config.dim, axis=-1)
    r = np.random.uniform()
    X_new_1 = Xb + self.__vb(t) * (W * XA - XB)
    X_new_2 = self.__vc(t) * X
    X_new = np.select([r < p, r >= p], [X_new_1, X_new_2])

    DF = bF if bF > DF else DF
    return X_new.clip(self.config.lower_bound, self.config.upper_bound), W_new, DF

  def run(self) -> MetricsType:
    X = np.random.normal(0, 1, (self.config.pop_size, self.config.dim)).clip(self.config.lower_bound, self.config.upper_bound) # locations/population/slime mould
    W = np.random.normal(0, 1, (self.config.pop_size, 1)) # weights
    DF = np.max(self.__fitness(X), 0)[0] # scalar

    for t in range(self.config.max_iters):
      X, W, DF = self.step(t, X.copy(), W.copy(), DF)
      
      if t % 10 == 0:
        id = np.argmax(self.__fitness(X), 0)[0]
        print(f"[Step {t}] DF: {DF}, X: {X[id]}")


class SlimeMouldSA:
  def __init__(self, objective_fn: Callable, config: Config) -> None:
    self.config = config
    self.objective_fn = objective_fn

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

  def step(self, t, X, W, DF, temp):
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
    p = np.repeat(p, self.config.dim, axis=-1)
    r = np.random.uniform()
    X_new_1 = Xb + self.__vb(t) * (W * XA - XB)
    X_new_2 = self.__vc(t) * X
    X_new = np.select([r < p, r >= p], [X_new_1, X_new_2])
    F_new = self.__fitness(X_new)
    X_explore = X_new + np.random.uniform(-0.5, 0.5, size=(self.config.pop_size, self.config.dim))
    F_explore = self.__fitness(X_explore)
    acceptance_probability = np.exp((F_explore - F_new) / np.clip(temp, 1e-5, np.infty)) # simulated annealing part
    # print(f"[acceptance_probability]: {acceptance_probability.mean()}")
    r2 = np.random.uniform(size = (self.config.pop_size, 1))
    pos1 = np.repeat(F_explore > F_new, self.config.dim, axis=-1)
    pos2 = np.repeat(np.any(np.stack([F_explore <= F_new, r2 < acceptance_probability], 0), 0), self.config.dim, axis=-1)
    neg = np.repeat(F_explore <= F_new, self.config.dim, axis=-1)
    X_new_new = np.select([pos1, pos2, neg], [X_explore, X_explore, X_new])

    # DF = bF if bF > DF else DF
    DF = np.max(self.__fitness(X_new_new), 0)[0]

    return X_new_new.clip(self.config.lower_bound, self.config.upper_bound), W_new, DF, temp * self.config.cooling_rate

  def run(self) -> MetricsType:
    X = np.random.normal(0, 1, (self.config.pop_size, self.config.dim)).clip(self.config.lower_bound, self.config.upper_bound) # locations/population/slime mould
    W = np.random.normal(0, 1, (self.config.pop_size, 1)) # weights
    DF = np.max(self.__fitness(X), 0)[0] # scalar
    temp = self.config.get("initial_temperature", 10)
    for t in range(self.config.max_iters):
      X, W, DF, temp = self.step(t, X.copy(), W.copy(), DF, temp)
      if t % 10 == 0:
        id = np.argmax(self.__fitness(X), 0)[0]
        print(f"[Step {t}] DF: {DF}, X: {X[id]}, temp: {temp}")

