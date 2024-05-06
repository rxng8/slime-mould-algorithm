# %%

import numpy as np
import matplotlib.pyplot as plt
import argparse

def bohachevsky(x: np.ndarray) -> np.ndarray:
  """Return the Bohachevsky function
  −100 ≤ xi ≤ 100
  The global minimum is located at x∗ = (0, 0) with f (x∗) = 0

  Args:
      x (np.ndarray): Shape (*B, D), any number of batch dimensions, and last dimension is the dim

  Returns:
      np.ndarray: (*B, 1)
  """
  x1 = x[..., :1]
  x2 = x[..., 1:2]
  return x1**2 + 2 * x2**2 - 0.3 * np.cos(3 * np.pi * x1) - 0.4 * np.cos(4 * np.pi * x2) + 0.7
bohachevsky.minimizing = True



def bird(x: np.ndarray) -> np.ndarray:
  """Return the Bird function
  −2pi ≤ xi ≤ 2pi
  The global minimum is located at x∗ = (4.70104, 3.15294)/(−1.58214, −3.13024) with f (x∗) = −106.764537

  Args:
      x (np.ndarray): Shape (*B, D), any number of batch dimensions, and last dimension is the dim

  Returns:
      np.ndarray: (*B, 1)
  """
  x1 = x[..., :1]
  x2 = x[..., 1:2]
  return np.sin(x1) * np.exp((1 - np.cos(x2))**2) + np.cos(x2) * np.exp((1 - np.sin(x1))**2) + (x1 - x2)**2
bird.minimizing = True




def bartelsconn(x: np.ndarray) -> np.ndarray:
  """Return the Bartels Conn function
  −500 ≤ xi ≤ 500
  The global minimum is located at x∗ = (0, 0) with f (x∗) = 1

  Args:
      x (np.ndarray): Shape (*B, D), any number of batch dimensions, and last dimension is the dim

  Returns:
      np.ndarray: (*B, 1)
  """
  x1 = x[..., :1]
  x2 = x[..., 1:2]
  return np.abs(x1**2 + x1*x2 + x2**2) + np.abs(np.sin(x1)) + np.abs(np.cos(x2))
bartelsconn.minimizing = True



def booth(x: np.ndarray) -> np.ndarray:
  """Return the Booth function
  −10 ≤ xi ≤ 10
  The global minimum is located at x∗ = (1, 3) with f (x∗) = 0

  Args:
      x (np.ndarray): Shape (*B, D), any number of batch dimensions, and last dimension is the dim

  Returns:
      np.ndarray: (*B, 1)
  """
  x1 = x[..., :1]
  x2 = x[..., 1:2]
  return (x1 + 2*x2 - 7)**2 + (2*x1 + x2 - 5)**2
booth.minimizing = True



def brent(x: np.ndarray) -> np.ndarray:
  """Return the Brent function
  −10 ≤ xi ≤ 10
  The global minimum is located at x∗ = (0, 0) with f (x∗) = 0

  Args:
      x (np.ndarray): Shape (*B, D), any number of batch dimensions, and last dimension is the dim

  Returns:
      np.ndarray: (*B, 1)
  """
  x1 = x[..., :1]
  x2 = x[..., 1:2]
  return (x1 + 10)**2 + (x2 + 10)**2 + np.exp(-(x1 ** 2 + x2 ** 2))
brent.minimizing = True




def beale(x: np.ndarray) -> np.ndarray:
  """Return the Beale function
  −4.5 ≤ xi ≤ 4.5
  The global minimum is located at x∗ = (3, 0.5) with f (x∗) = 0

  Args:
      x (np.ndarray): Shape (*B, D), any number of batch dimensions, and last dimension is the dim

  Returns:
      np.ndarray: (*B, 1)
  """
  x1 = x[..., :1]
  x2 = x[..., 1:2]
  return (0.5 - x1 + x1 * x2)**2 + (2.25 - x1 + x1 * x2**2)**2 + (2.625 - x1 + x1 * x2**3)**2
beale.minimizing = True



def camel(x: np.ndarray) -> np.ndarray:
  """Return the Camel function
  −5 ≤ xi ≤ 5
  The global minimum is located at x∗ = (0, 0) with f (x∗) = 0

  Args:
      x (np.ndarray): Shape (*B, D), any number of batch dimensions, and last dimension is the dim

  Returns:
      np.ndarray: (*B, 1)
  """
  x1 = x[..., :1]
  x2 = x[..., 1:2]
  return 2 * x1**2 - 1.05 * x1**4 + (1/6) * x1**6 + x1*x2 + x2**2
camel.minimizing = True



def bukin(x: np.ndarray) -> np.ndarray:
  """Return the Bukin function
  −15 ≤ x1 ≤ -5
  −3 ≤ x2 ≤ −3
  The global minimum is located at x∗ = (−10, 0) with f (x∗) = 0

  Args:
      x (np.ndarray): Shape (*B, D), any number of batch dimensions, and last dimension is the dim

  Returns:
      np.ndarray: (*B, 1)
  """
  x1 = x[..., :1]
  x2 = x[..., 1:2]
  return 100 * (x2 - 0.01 * x1**2 + 1) + 0.01 * (x1 + 10)**2
bukin.minimizing = True



def cube(x: np.ndarray) -> np.ndarray:
  """Return the Cube function
  −10 ≤ xi ≤ 10.
  The global minimum is located at x∗ = (−1, 1) with f (x∗) = 0

  Args:
      x (np.ndarray): Shape (*B, D), any number of batch dimensions, and last dimension is the dim

  Returns:
      np.ndarray: (*B, 1)
  """
  x1 = x[..., :1]
  x2 = x[..., 1:2]
  return 100 * (x2 - x1**3)**2 + (1 - x1)**2
cube.minimizing = True



def negative_Alpine(x: np.ndarray) -> np.ndarray:
  """Return the negative Alpine function

  Args:
      x (np.ndarray): Shape (*B, D), any number of batch dimensions, and last dimension is the dim

  Returns:
      np.ndarray: (*B, 1)
  """
  return -np.abs((x * np.sin(x) + 0.1 * x)).sum(-1, keepdims=True)
negative_Alpine.minimizing = False


def rosenbrock(x: np.ndarray) -> np.ndarray:
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
rosenbrock.minimizing = True


def easom(x: np.ndarray) -> np.ndarray:
  """Return the Easom function

  Args:
      x (np.ndarray): Shape (*B, D), any number of batch dimensions, and last dimension is the dim

  Returns:
      np.ndarray: (*B, 1)
  """
  x1 = x[..., :1]
  x2 = x[..., 1:2]
  return - np.cos(x1) * np.cos(x2) * np.exp(- (x1 - np.pi)**2 - (x2 - np.pi)**2)
easom.minimizing = True


def fourpeak(x):
  """Return the Four-peak function

  Args:
      x (np.ndarray): Shape (*B, D), any number of batch dimensions, and last dimension is the dim

  Returns:
      np.ndarray: (*B, 1)
  """
  x1 = x[..., :1]
  x2 = x[..., 1:2]
  return (np.exp(-((x1 - 4) ** 2 + (x2 - 4) ** 2)) + np.exp(-((x1 + 4) ** 2 + (x2 - 4) ** 2)) + 2 * (
          np.exp(-(x1 ** 2 + x2 ** 2)) + np.exp(-(x1 ** 2 + (x2 + 4) ** 2))))
fourpeak.minimizing = False



def eggcrate(x):
  """Return the Eggcrate function

  Args:
      x (np.ndarray): Shape (*B, D), any number of batch dimensions, and last dimension is the dim

  Returns:
      np.ndarray: (*B, 1)
  """
  x1 = x[..., :1]
  x2 = x[..., 1:2]
  return - (x1 ** 2 + x2 ** 2 + 25 * (np.sin(x1) ** 2 + np.sin(x2) ** 2 ))
eggcrate.minimizing = True



def ackley(x: np.ndarray) -> np.ndarray: # returns a scalar
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
ackley.minimizing = True



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
        'neg_alpine', 'rosenbrock', 'ackley', 'bukin', 'easom', 'bohachevsky', 
        'bird', 'bartelsconn', 'booth', 'brent', 'beale', 'camel'])
    args = parser.parse_args()
    
    if args.fn == 'ackley':
        plot_func(ackley, "Ackley's Function")
    elif args.fn == 'neg_alpine':
        plot_func(negative_Alpine, "Negative Alpine Function")
    elif args.fn == 'rosenbrock':
        plot_func(rosenbrock, "Rosenbrock Function")
    elif args.fn == 'bukin':
        plot_func(bukin, "Bukin")
    elif args.fn == 'easom':
        plot_func(easom, "Easom")
    elif args.fn == 'bohachevsky':
        plot_func(bohachevsky, "Bohachevsky")
    elif args.fn == 'bird':
        plot_func(bird, "Bird")
    elif args.fn == 'bartelsconn':
        plot_func(bartelsconn, "Bartelsonn")
    elif args.fn == 'booth':
        plot_func(booth, 'Booth')
    elif args.fn == 'brent':
        plot_func(brent, 'Brent')
    elif args.fn == 'beale':
        plot_func(beale, 'Beale')
    elif args.fn == 'camel':
        plot_func(camel, 'Camel')
    else:
        print(f"Err 404: function {args.fn} not found.")
        

if __name__ == "__main__":
    main()
