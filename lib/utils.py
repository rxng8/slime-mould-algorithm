# %%
import os
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
    criteria_config = config['stop_criterion']
    if 'criteria' in criteria_config:
        # Handle multiple criteria
        criteria = []
        for criterion in criteria_config['criteria']:
            if criterion['type'] == 'temperature':
                min_temperature = criterion.get('min_temperature', float('inf'))
                criteria.append(lambda state, temp=min_temperature: stop_by_temperature(state, temp))
            elif criterion['type'] == 'iterations':
                max_iterations = criterion.get('max_iterations', 1000)
                criteria.append(lambda state, max_i=max_iterations: stop_by_iterations(state, max_i))
            elif criterion['type'] == 'fitness':
                target_fitness = criterion.get('target_fitness', float('inf'))
                criteria.append(lambda state, fit=target_fitness: stop_by_fitness(state, fit))
        
        # Combine criteria: stop if any criterion returns True
        return lambda state: any(crit(state) for crit in criteria)
    else:
        if criteria_config['type'] == 'temperature':
            return lambda state: stop_by_temperature(state, criteria_config.get('min_temperature', float('inf')))
        elif criteria_config['type'] == 'iterations':
            return lambda state: stop_by_iterations(state, criteria_config.get('max_iterations', 1000))
        elif criteria_config['type'] == 'fitness':
            return lambda state: stop_by_fitness(state, criteria_config.get('target_fitness', float('inf')))



def generate_latex_table(results: Dict[int, List[str]], headers: List[str], experiment_name: str):
    """Given all results from the experiment, it creates a table in LaTex format for the report.
    Args:
        results (Dict[int, List[str]]): List of a dict(key: population size, value: mean / std)
        headers (List[str]): Header nformation of the table
        file_path (str): The path where to save latex file
    """
    file_path = os.path.join('experiment', experiment_name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure the directory exists
    
    # File path for the LaTeX file
    file_path = os.path.join(file_path, 'results_table.tex')
    
    with open(file_path, 'w') as file:
        file.write('\\begin{tabularx}{\\textwidth}{|' + 'X|' * len(headers) + '}\n\\hline\n')
        header_row = ' & '.join(headers) + ' \\\\\n\\hline\n'
        file.write(header_row)
        for key, values in results.items():
            row = f'$n = {key}$ & ' + ' & '.join(values) + ' \\\\\n'
            file.write(row)
        file.write('\\hline\n\\end{tabularx}\n')