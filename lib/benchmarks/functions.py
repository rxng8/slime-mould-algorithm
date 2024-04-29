# %%

import numpy as np
import matplotlib.pyplot as plt
import argparse

def negative_Alpine(x: np.ndarray) -> np.ndarray:
  """Return the negative Alpine function

  Args:
      x (np.ndarray): Shape (*B, D), any number of batch dimensions, and last dimension is the dim

  Returns:
      np.ndarray: (*B, 1)
  """
  return -np.abs((x * np.sin(x) + 0.1 * x)).sum(-1, keepdims=True)

def Rosenbrock(x: np.ndarray) -> np.ndarray:
  """Return the negative Rosenbrock function

  Args:
      x (np.ndarray): Shape (*B, D), any number of batch dimensions, and last dimension is the dim

  Returns:
      np.ndarray: (*B, 1)
  """
  # x: (D)
  x_i = x[:-1]
  x_ip1 = x[1:]
  # out: (1)
  return ((x_i - 1)**2 + 100 * ((x_ip1 - x_i**2))**2).sum(-1, keepdims=True)







def easom(x: np.ndarray) -> np.ndarray:
  """Return the Easom function

  Args:
      x (np.ndarray): Shape (*B, D), any number of batch dimensions, and last dimension is the dim

  Returns:
      np.ndarray: (*B, 1)
  """
  x1, x2 = x.T
  return - np.cos(x1) * np.cos(x2) * np.exp(- (x1 - np.pi)**2 - - (x2 - np.pi)**2)



def fourpeak(x):
  """Return the Four-peak function

  Args:
      x (np.ndarray): Shape (*B, D), any number of batch dimensions, and last dimension is the dim

  Returns:
      np.ndarray: (*B, 1)
  """
  x1, x2 = x.T
  return (np.exp(-((x1 - 4) ** 2 + (x2 - 4) ** 2)) + np.exp(-((x1 + 4) ** 2 + (x2 - 4) ** 2)) + 2 * (
          np.exp(-(x1 ** 2 + x2 ** 2)) + np.exp(-(x1 ** 2 + (x2 + 4) ** 2))))



def eggcrate(x):
  """Return the Eggcrate function

  Args:
      x (np.ndarray): Shape (*B, D), any number of batch dimensions, and last dimension is the dim

  Returns:
      np.ndarray: (*B, 1)
  """
  x1, x2 = x.T
  return - (x1 ** 2 + x2 ** 2 + 25 * (np.sin(x1) ** 2 + np.sin(x2) ** 2 ))




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


def plot_func(f, name:str="function") -> None:
    """Creates a visula of a objectve function using matplotlib.
    Args:
        f (function): The objective function.
        name (str, optional): The title of the plot. Defaults to "function".
    """
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    fn = np.zeros(X.shape)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            fn[i, j] = f(np.array([X[i, j], Y[i, j]]))

    fig = plt.figure(figsize=(8, 6))

    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, fn, cmap='viridis')
    ax.set_title(name)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark functions.")
    parser.add_argument("--fn", help="Specify function to plot", choices=[
        'neg_alpine', 'rosenbrock', 'ackley'])
    args = parser.parse_args()
    
    if args.fn == 'ackley':
        plot_func(Ackley, "Ackley's Function")
    elif args.fn == 'neg_alpine':
        plot_func(negative_Alpine, "Negative Alpine Function")
    elif args.fn == 'rosenbrock':
        plot_func(Rosenbrock, "Rosenbrock Function")
    else:
        print(f"Err 404: function {args.fn} not found.")
        

if __name__ == "__main__":
    main()
