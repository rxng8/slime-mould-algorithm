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


def test(x):
    return np.sum( 1 / x)


def plot_func(f, name:str="function") -> None:
    """Creates a visula of a objectve function using matplotlib.
    Args:
        f (function): The objective function.
        name (str, optional): The title of the plot. Defaults to "function".
    """
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
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
        'neg_alpine', 'rosenbrock', 'ackley', 'test'])
    args = parser.parse_args()
    
    if args.fn == 'ackley':
        plot_func(Ackley, "Ackley's Function")
    elif args.fn == 'neg_alpine':
        plot_func(negative_Alpine, "Negative Alpine Function")
    elif args.fn == 'rosenbrock':
        plot_func(Rosenbrock, "Rosenbrock Function")
    elif args.fn == 'test':
        plot_func(test)
    else:
        print(f"Err 404: function {args.fn} not found.")
        

if __name__ == "__main__":
    main()