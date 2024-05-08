# Slime Mould Algorithm

Group member: Viet, Matthias, Faeze


## How to Setup an Experiment
This library provides an environment to experiment with a selection methaheristic algorithms on 13 different benchmark functions.<br>
Experiments can be setup as stand alone algorithm performance evaluation and hyperparameter fine-tuning on a benchmark function, 
or as a performance comparision of multiple methaheuristic algorithms. 

### Create a `experinement.py` file with the following setup. (stand alone)

#### Imports and Global Variables

##### 1. Choose Algorithm and Benchmark
```Python
from lib.algorithms import SimulatedAnnealing
from lib.benchmarks import Rosenbrock
```
- Algorithm support for: `SlimeMold`, `SimulatedAnnealing`, `FireFly`, `DifferentialEvolution`, `SimulatedAnnealingSlimeMould`, `ParticleSwarm`, `FireFly`, `Bat`
- Benchmark support for: `bohachevsky`, `bird`, `bartelsconn`, `booth`, `brent`, `beale`, `camel`, `bukin`, `cube`, `negative_Alpine`, `rosenbrock`, `easom`, `fourpeak`, `eggcrate`, `ackley`

functions can be visualized through `lib\benchmarks\functions.py`.
```
functions.py -h
usage: functions.py [-h]
                    [--fn {neg_alpine,rosenbrock,ackley,bukin,easom,bohachevsky,bird,bartelsconn,booth,brent,beale,camel,fourpeak}]

Plot benchmark functions.

options:
  -h, --help            show this help message and exit
  --fn {neg_alpine,rosenbrock,ackley,bukin,easom,bohachevsky,bird,bartelsconn,booth,brent,beale,camel,fourpeak}       
                        Specify function to plot
```

##### 2. Add default library support
```Python
from lib import Config  # Custom Object holding parameters
from lib.solve import solve # solve wrapper
```

##### 3. Define Global Experiment Variables
- `SEED`: Seed random function for deterministic outcome
- `D`: Dimensions of search space of benchmark function
- `LB`: Lower bound of benchmark function
- `UB`: Upper bound of benchmark function
- `TRIALS`: Number of random trials for experiment

#### Define Algorithm parameters
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
A `Config` stores all required parameters in order to be able to execute the algorithm. 
In order to understand the setup for a specific algorithm please checkout the `example` folder with
example projects for every methaheuristc algorithm available in this library.

#### Run an Algorithm using `solve`
In order to run experiment, the `solve` wrapper function from `lib/solve.py/` has to be called.

__Parameters__:
- `trials`(int): The number of independent trials to run the specified algorithm. Each trial will start with possibly different initial conditions if the algorithm is stochastic.
- `Algo` (Callable): The optimization algorithm to be tested. This function should be capable of being called with a configuration object and should return the fitness score of the solution it computes.
- `config` (Callable): A configuration object or function that provides parameters needed by the Algo. The specifics of config should align with the requirements of the passed Algo.
- `log_to_file` (bool, optional): If True, logs will be written to a file. Defaults to False.
- `experiment_name` (str, optional): A name for the experiment, which will be used to name log files and directories. Defaults to an empty string, which results in using 'default_logger' as the logger name.

__Returns__:
- `fitness_scores` (numpy.ndarray): An array containing the fitness scores from each trial.
- `mean_fitness` (float): The mean fitness score across all trials.
- std_dev_fitness (float): The standard deviation of the fitness scores across all trials.
- `result_string` (str): A formatted string representing the mean and standard deviation of the fitness scores, suitable for presentation in scientific documents (e.g., LaTeX format).

__Example Usage__:
```python
from lib.algorithms import MyAlgorithm
from lib.config import MyConfig

# Configuration setup for the algorithm
config = MyConfig(param1=10, param2=0.5)

# Running the solve function
fitness_scores, mean_fitness, std_dev_fitness, result_string = solve(
    trials=30,
    Algo=MyAlgorithm,
    config=config,
    log_to_file=True,
    experiment_name='MyExperiment'
)
```

#### Example Experiment
In this example we are creating an experiment using the `FireFly` algorithm.

##### 1. __Import algorithm and other functions__
```python
import numpy as np
from typing import Dict, List

from lib.config import Config
from lib.algorithms import FireFly
from lib.benchmarks import Rosenbrock
from lib.solve import solve
from lib.utils import generate_latex_table
```
The code is performing an experiment on the `Rosenbrock` benchmark function provided in the library (Gloval Optima at `f(.) = 0`). The `Config` is a custom data structure stroing parameters required by the algorithm.

##### 2. __Set Global Variables__
```python
SEED = 1990
D = 2
LB = -2.0
UB = 2.0
TRIALS = 30
MAX_I = 1000

np.random.seed(SEED)
```

##### 3. __Populate the `Config`__
```python
config = Config(
    D=D,
    lb=LB,
    ub=UB,
    funct=Rosenbrock,
    stop_criterion={
        'type': 'complex',
        'criteria': [
            {'type': 'iterations', 'max_iterations': 1000},
            {'type': 'fitness', 'target_fitness': 0.01}
        ]
    },
    min="Minimization",
    gamma=0.0,
    alpha=0.0,
    beta0=0.0,
    population=0,
)
```
Note that the hyper-parameters are set to `0`. Since this experiment is running multiple settings, the initial value has to be set in will be overwritten in each experiment stage.

##### 4. __Define hyper-parameters__
```python
swarm_sizes = [5, 10, 25, 50, 100]
# different hyper-parameters used
hyperparam_list = [
    {'gamma': 1.0, 'alpha': 0.5, 'beta0':1.0},
    {'gamma': 1.0, 'alpha': 0.2, 'beta0':1.0},
    {'gamma': 1.0, 'alpha': 0.2, 'beta0':0.2}
]
```
As mentioned in __(3.)__, the set of used hyper-parameters has to be specified.

##### 5. __Define Structure for LaTeX export__
```python
results = {size: [] for size in swarm_sizes}
# Headers based on hyperparameter configurations
headers = ['Popul. Size'] + [f'$\\gamma={hp["gamma"]}, \\alpha={hp["alpha"]}, \\beta_0={hp["beta0"]}$' for hp in hyperparam_list]

# Experiment logging info
experiment_name = 'test_FireFly'
```

In order to be able to create a LaTeX table storing the computed values, the `header` of the table and the bins for the `mean` and `standard deviation` of each experiment has to be defined.<br>
An `experiment_name` is also needed to store loggings and the LaTeX table.

##### 6. __Run Algorithm__:
```python
for swarm_size in swarm_sizes:
    config = config.update(population=swarm_size)
    
    for hyper_params in hyperparam_list:
        config = config.update(hyper_params)
        # Run experiment
        _, mean_result, std_dev_result, latex_result = solve(TRIALS, FireFly, config, log_to_file=True, experiment_name=experiment_name)
        results[swarm_size].append(latex_result)
        
generate_latex_table(results, headers, experiment_name)
```

The algorithm is run for every specified `pop_size` in `pop_sizes`. For each different population size, each of the spefied `hyper_params` of the `hyperparam_list` is updated in `config` and the `solve` function is called. For the solve function, we need to add the number of `TRIALS` for each parameter setting, the `FireFly` algorithm, the `config` file, and the `experiment_name`. As a result we get the last `fitness_scores`, the `mean_result`, the `std_dev_result` (standard deviation of the trials), and a `latex_result` (results in LaTeX format).<br>
`generate_latex_table` creates the LaTeX table given a list of the LaTeX results, the header and the experiment name.

The full code of this example can be found in `run_FireFly.py`.

### Create a `compare.py` file with the following setup. (comparison analysis)
The following section explains how to run or setup a `compare.py`. The example shown here is the `run_comparison.py`.


#### Imports and Global Variables

```python
import numpy as np
from typing import Dict, List
from lib.config import Config
from lib.algorithms import FireFly, Bat, PSO, SlimeMould, SimulatedAnnealingSlimeMould
from lib.benchmarks.functions import ackley
import lib.benchmarks.functions as function_module
from inspect import getmembers, isfunction
from lib.solve import compare
```
- Select the __algorithms__ and __benchmark function__ to work with
- import the `compare` function from `lib.solve`

```python
SEED = 1990
    TRIALS = 30
    GLOBAL = {
        'D': 2,
        'lb': -35.0,
        'ub': 35.0,
        'MAX_I': 1000,
        'funct': FUNCT,
        'min' :"Minimization",
        'minimizing': FUNCT.minimizing,
        'stop_criterion': {
            'type': 'complex',
            'criteria': [
                {'type': 'iterations', 'max_iterations': 1000},
                {'type': 'fitness', 'target_fitness': 1e-6}
            ]
        }
    }

    np.random.seed(SEED)

    populations = [5, 10, 25, 50, 100, 500]
```
Create a `GLOBAL` dictionay that holds all constant variables through all the algorithms compared against. In this example the population size is chhanged through multiple trials handled by the `solve()` wrapper. This is currently the only variable parameter in the comparison farmework.

```python
firefly_config = Config(gamma=1.0, alpha=0.5, beta0=1.0)
firefly_config = firefly_config.update(GLOBAL)
```
Create a `Config` struct for each algorithm that holds it's specific parameters.

```python
configs = [firefly_config, bat_config, pso_config, slime_mould_config, simulated_annealing_slime_mould_config]
algos = [FireFly, Bat, PSO, SlimeMould, SimulatedAnnealingSlimeMould]

compare(algos, configs, populations, TRIALS, f"{func_name}")
```
Create lists of each algorithm and it's corresponding `Config` and call `compare`. This generates a folder of the compre experiment started, including a log file (or print to terminal) and a LaTex table with the results of the experiment. Additionally a time per trial plot can be generated through `config(plot_time=True)`.
