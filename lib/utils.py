# %%

from typing import List, Dict, Tuple, Callable, Sequence, Any
import numpy as np

def get_initializer(name: str) -> Callable:
    return {
        "normal": lambda pop_size, dim: np.random.normal((pop_size, dim)),
        "zeros": lambda pop_size, dim: np.zeros((pop_size, dim)),
        "uniform": lambda pop_size, dim, lb, ub: np.random.uniform(lb, ub, (pop_size, dim))
    }[name]
    
    
def stop_by_temperature(state, min_temperature):
    return state.get('temperature', float('inf')) <= min_temperature


def stop_by_iterations(state, max_iterations):
    return state.get('iterations', 0) >= max_iterations


def stop_by_fitness(state, target_fitness):
    return state.get('current_fitness', float('inf')) <= target_fitness
    

def define_stop_criterion(config):
    criterion_type = config['stop_criterion']['type']
    if criterion_type == 'temperature':
        min_temperature = config['stop_criterion'].get('min_temperature', 0)
        return lambda state: stop_by_temperature(state, min_temperature)
    elif criterion_type == 'iterations':
        max_iterations = config['stop_criterion'].get('max_iterations', 1000)
        return lambda state: stop_by_iterations(state, max_iterations)
    elif criterion_type == 'fitness':
        target_fitness = config['stop_criterion'].get('target_fitness', float('inf'))
        return lambda state: stop_by_fitness(state, target_fitness)
    