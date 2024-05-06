
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
    minimizing = self.config.minimizing
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
    mid = self.config.population // 2
    W_new = W
    W_new[:mid] = 1 - tricky_term[:mid]
    W_new[mid:] = 1 + tricky_term[mid:]

    ### Wrap food and Oscilation
    iA, iB = np.random.choice(self.config.population, 2, replace=False)
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
    X = np.random.normal(0, 1, (self.config.population, self.config.D)).clip(self.config.lb, self.config.ub) # locations/population/slime mould
    W = np.random.normal(0, 1, (self.config.population, 1)) # weights

    curr_best_id = np.argmax(self.__fitness(X), axis=0)[0]
    X_gstar = X[curr_best_id]
    DF = self.__fitness(X_gstar)[0] # scalar
    state = {'current_fitness': -DF if self.config.minimizing else DF}

    for t in range(self.config.max_iters):
      if self.stop(state):
        break
      X, W, DF = self.step(t, X.copy(), W.copy(), DF)
      # if t % 10 == 0:
        # id = np.argmax(self.__fitness(X), 0)[0]
        # print(f"[Step {t}] X: {X[id]}, best: {self.__fitness(X[id])[0]}")
      curr_best_id = np.argmax(self.__fitness(X), axis=0)[0]
      X_curr_best = X[curr_best_id]
      if self.__fitness(X_curr_best) > self.__fitness(X_gstar):
        X_gstar = X_curr_best.copy()

      state.update({'current_fitness': -DF if self.config.minimizing else DF})

    # id = np.argmax(self.__fitness(X), 0)[0]
    # return X[id], -self.__fitness(X[id])[0] if self.config.minimizing else self.__fitness(X[id])[0]

    F = self.__fitness(X_gstar)[0]
    return X_gstar, -F if self.config.minimizing else F


class SimulatedAnnealingSlimeMould:
  def __init__(self, config: Config) -> None:
    self.config = config
    self.objective_fn = config.funct
    self.stop: Callable = define_stop_criterion(config)

  def __fitness(self, X):
    # This algorithm is defaulted to maximizing the objective function
    minimizing = self.config.minimizing
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
    mid = self.config.population // 2
    W_new = W
    W_new[:mid] = 1 - tricky_term[:mid]
    W_new[mid:] = 1 + tricky_term[mid:]

    ### Wrap food and Oscilation
    iA, iB = np.random.choice(self.config.population, 2, replace=False)
    XA, XB = X[iA][None], X[iB][None] # (1, D)
    Xb = X[0][None] # (1, D)
    p = np.tanh(np.abs(F - DF)) # (B, 1)
    p = np.repeat(p, self.config.D, axis=-1)
    r = np.random.uniform()
    X_new_1 = Xb + self.__vb(t) * (W * XA - XB)
    X_new_2 = self.__vc(t) * X
    X_new = np.select([r < p, r >= p], [X_new_1, X_new_2])
    F_new = self.__fitness(X_new)
    X_explore = X_new + np.random.uniform(-0.5, 0.5, size=(self.config.population, self.config.D))
    F_explore = self.__fitness(X_explore)
    acceptance_probability = np.exp((F_explore - F_new) / np.clip(temp, 1e-5, np.infty)) # simulated annealing part
    # print(f"[acceptance_probability]: {acceptance_probability.mean()}")
    r2 = np.random.uniform(size = (self.config.population, 1))
    pos1 = np.repeat(F_explore > F_new, self.config.D, axis=-1)
    pos2 = np.repeat(np.any(np.stack([F_explore <= F_new, r2 < acceptance_probability], 0), 0), self.config.D, axis=-1)
    neg = np.repeat(F_explore <= F_new, self.config.D, axis=-1)
    X_new_new = np.select([pos1, pos2, neg], [X_explore, X_explore, X_new])

    # DF = bF if bF > DF else DF
    DF = np.max(self.__fitness(X_new_new), 0)[0]

    return X_new_new.clip(self.config.lb, self.config.ub), W_new, DF, temp * self.config.cooling_rate

  def solve(self) -> tuple:
    X = np.random.normal(0, 1, (self.config.population, self.config.D)).clip(self.config.lb, self.config.ub) # locations/population/slime mould
    W = np.random.normal(0, 1, (self.config.population, 1)) # weights

    curr_best_id = np.argmax(self.__fitness(X), axis=0)[0]
    X_gstar = X[curr_best_id]
    DF = self.__fitness(X_gstar)[0] # scalar
    temp = self.config.get("initial_temperature", 10)

    state = {'current_fitness': -DF if self.config.minimizing else DF}

    for t in range(self.config.max_iters):
      if self.stop(state):
        break
      X, W, DF, temp = self.step(t, X.copy(), W.copy(), DF, temp)
      curr_best_id = np.argmax(self.__fitness(X), axis=0)[0]
      X_curr_best = X[curr_best_id]
      if self.__fitness(X_curr_best) > self.__fitness(X_gstar):
        X_gstar = X_curr_best.copy()

      state.update({'current_fitness': -DF if self.config.minimizing else DF})

    # id = np.argmax(self.__fitness(X), 0)[0]
    # return X[id], -self.__fitness(X[id])[0] if self.config.minimizing else self.__fitness(X[id])[0]

    F = self.__fitness(X_gstar)[0]
    return X_gstar, -F if self.config.minimizing else F
