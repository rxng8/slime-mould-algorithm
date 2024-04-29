# Slime Mould Algorithm

Group member: Ugly-Viet, Matthias, Faeze


## How to Setup an Experiment

Create a `experinement.py` file with the following setup.

### Imports and Global Variables

1. Choose Algorithm and Benchmark
```Python
from lib.algorithms import SimulatedAnnealing
from lib.benchmarks import Rosenbrock
```
- Algorithm support for: `SlimeMold`, `SimulatedAnnealing`, `FireFly`, `DifferentialEvolution`
- Benchmark support for: ...

2. Add default library support
```Python
from lib import Config  # Custom Object holding parameters
from lib.solve import solve # solve wrapper
```

3. Define Global Experiment Variables
- `SEED`: Seed random function for deterministic outcome
- `D`: Dimensions of search space of benchmark function
- `LB`: Lower bound of benchmark function
- `UB`: Upper bound of benchmark function
- `TRIALS`: Number of random trials for experiment

### Define Algorithm parameters
Each algorithm has its own hyper-parameters. All required variables need to be added as field to `Config`. Also add the global environment setting to this object.

```Python
# Example for Simmulated Annealing
config = Config(
    dim = D,
    lb = LB,
    ub = UB,
    initializer = "uniform", # Define random init Agents
    cooling = "geometric", # Define cooling schedule
    T_0 = 100,
    T_f = 1e-10,
    t_f = 1000,
    alpha = 0.1,
    a = 0.99,
    f = Rosenbrock
)
```
