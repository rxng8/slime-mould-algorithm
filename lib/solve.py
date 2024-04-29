import logging, os
from typing import Callable

# External
import numpy as np

def setup_logger(name: str, log_file: str = None, level: int = logging.INFO, stream: bool = True) -> logging.Logger:
    """Setup and return a configured logger."""
    logger = logging.getLogger(name)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    logger.setLevel(level)
    return logger


def create_experiment_directory(base_path: str, experiment_name: str) -> str:
    """Ensure the experiment directory exists and return its path."""
    experiment_dir = os.path.join(base_path, '..', 'experiment', experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir


def run_experiment(trials: int, Algo: Callable, config: Callable, logger: logging.Logger):
    """Run the experiment and collect fitness scores."""
    fitness_scores = []
    for trial in range(trials):
        algo = Algo(config)
        position, fitness = algo.solve()
        fitness_scores.append(fitness)
        logger.info(f"Trial {trial + 1}: Position = {position}, Fitness = {fitness}")
    return np.array(fitness_scores)


def solve(trials: int, Algo: Callable, config: Callable, log_to_file: bool = False, experiment_name: str = ''):
    """A function to perform an experiment and log the results."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    experiment_dir = create_experiment_directory(current_dir, experiment_name)
    
    logger_name = experiment_name if experiment_name else 'default_logger'
    log_file_path = os.path.join(experiment_dir, f'{logger_name}.log') if log_to_file else None
    logger = setup_logger(logger_name, log_file_path, stream=not log_to_file)
    
    logger.info(f"Configuration: {vars(config)}")
    fitness_scores = run_experiment(trials, Algo, config, logger)
    
    mean_fitness = np.mean(fitness_scores)
    std_dev_fitness = np.std(fitness_scores)
    
    logger.info(f"Mean of fitness scores: {mean_fitness}")
    logger.info(f"Standard Deviation of fitness scores: {std_dev_fitness}")
    
    result_string = f"${mean_fitness:.7g} \\pm {std_dev_fitness:.7g}$"
    return fitness_scores, mean_fitness, std_dev_fitness, result_string