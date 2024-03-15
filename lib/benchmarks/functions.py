# %%

import numpy as np

def negative_Alpine(x: np.ndarray) -> np.ndarray:
  """Return the negative Alpine function

  Args:
      x (np.ndarray): Shape (*B, D), any number of batch dimensions, and last dimension is the dim

  Returns:
      np.ndarray: (*B, 1)
  """
  return -np.abs((x * np.sin(x) + 0.1 * x)).sum(-1, keepdims=True)
negative_Alpine.minimizing = False

def Alpine(x: np.ndarray) -> np.ndarray:
  """Return the negative Alpine function

  Args:
      x (np.ndarray): Shape (*B, D), any number of batch dimensions, and last dimension is the dim

  Returns:
      np.ndarray: (*B, 1)
  """
  return np.abs((x * np.sin(x) + 0.1 * x)).sum(-1, keepdims=True)
Alpine.minimizing = True

def Rosenbrock(x: np.ndarray) -> np.ndarray:
  """Return the negative Rosenbrock function

  Args:
      x (np.ndarray): Shape (*B, D), any number of batch dimensions, and last dimension is the dim

  Returns:
      np.ndarray: (*B, 1)
  """
  # x: (D)
  x_i = x[..., :-1]
  x_ip1 = x[..., 1:]
  # out: (1)
  return ((x_i - 1)**2 + 100 * ((x_ip1 - x_i**2))**2).sum(-1, keepdims=True)
Rosenbrock.minimizing = True

def Ackley(x: np.ndarray) -> np.ndarray: # returns a scalar
  """Return the Ackley function

  Args:
      x (np.ndarray): Shape (*B, D), any number of batch dimensions, and last dimension is the dim

  Returns:
      np.ndarray: (*B, 1)
  """
  d = x.shape[-1]
  d_inv = 1.0 / d
  a = -20 * np.exp(-0.02 * np.sqrt(d_inv * (x**2).sum(-1, keepdims=True)))
  b = np.exp(d_inv * np.cos(2 * np.pi * x).sum(-1, keepdims=True))
  return a - b + 20 + np.e
Ackley.minimizing = True

