import numpy as np
from typing import Callable
from lib.utils import define_stop_criterion

class Bat:
    def __init__(self, config: object) -> None:
        """
        Initialize the Bat algorithm with a configuration object.
        Parameters:
            config: An object containing all necessary settings, including:
                - D (dimension of the problem)
                - funct (objective function to optimize)
                - min (boolean indicating whether the problem is minimization)
                - max_iter_no_improv (maximum iterations without improvement for stopping)
                - max_i (maximum iterations)
                - lb, ub (lower and upper bounds for the search space)
                - population (number of bats)
                - beta0, gamma, alpha (algorithm parameters for movement, frequency, and amplitude)
        """
        # Optimization Problem
        self.D = config.D
        self.f = config.funct
        
        # Minimization or Maximization Problem
        self.min = True if hasattr(config, "min") else False
        
        # Stop Criterion
        self.stop: Callable = define_stop_criterion(config)
        
        # constraints
        self.freq_max = config.freq_max
        self.freq_min = config.freq_min
        self.alpha = config.alpha
        self.init_pulse_rate = config.pulse_rate
        self.gamma = config.gamma
        
        # bat
        self.bat_pos = np.random.uniform(config.lb, config.ub, size=(config.population, config.D))
        self.velocity = np.zeros((config.population, config.D))
        # randomly initialize a freuency (different to book)
        # self.frequency = np.random.uniform(config.freq_min, config.freq_max, config.population)
        self.frequency = np.zeros((config.population,))

        # self.frequency = np.zeros((config.))
        self.loudness = np.full(config.population, config.loudness) # constant in book?
        self.pulse_rate = np.full(config.population, config.pulse_rate) # constant in book?
        self.fitness = self._get_fitness().flatten() # (N,)

        # Bounds
        self.ub = config.ub
        self.lb = config.lb
    
    def _get_fitness(self):
        """
        Evaluate the fitness of all bats using the objective function.
        Returns:
            An array of fitness values for each bat. shape(N, 1)
        """
        objective = np.apply_along_axis(self.f, 1, self.bat_pos)
        return objective if self.min else -objective

    def _update_bats(self):
        """
        Update the positions and velocities of bats based on their frequencies, the best bat's position, and a random walk if conditions are met.
        Implements local search and velocity updates.
        """
        # Find the best bat index
        best_bat_idx = np.argmin(self.fitness) if self.min else np.argmax(self.fitness)
        best_bat_pos = self.bat_pos[best_bat_idx]

        # Update frequencies using random betas
        betas = np.random.rand(len(self.bat_pos))
        self.frequency = self.freq_min + (self.freq_max - self.freq_min) * betas

        # Update velocities
        self.velocity += (self.bat_pos - best_bat_pos) * self.frequency[:, None]

        # Candidate positions
        candidate_pos = self.bat_pos + self.velocity

        # Random walk condition
        random_walk_mask = np.random.rand(len(self.bat_pos)) > self.pulse_rate
        random_walk_positions = best_bat_pos + 0.01 * np.random.randn(len(self.bat_pos), self.D)
        candidate_pos[random_walk_mask, :] = random_walk_positions[random_walk_mask, :]

        # Evaluate candidate fitness
        candidate_fitness = np.apply_along_axis(self.f, 1, candidate_pos).flatten() # (N,)

        # Apply selection criteria
        acceptance_mask = (np.random.rand(len(self.bat_pos)) < self.loudness) & (candidate_fitness < self.fitness) # (N,)

        self.bat_pos[acceptance_mask, :] = np.clip(candidate_pos[acceptance_mask, :], self.lb, self.ub)
        self.fitness[acceptance_mask] = candidate_fitness[acceptance_mask]
        self.loudness[acceptance_mask] *= self.alpha
        self.pulse_rate[acceptance_mask] = self.init_pulse_rate * (1 - np.exp(self.gamma * np.arange(len(self.bat_pos))[acceptance_mask]))
    
    
    def _update_fitness(self):
        """
        Update fitness values for all bats and check for improvements.
        Returns:
            True if any bat's fitness has improved, otherwise False.
        """
        new_fitness = self._get_fitness().flatten()  # Flatten fitness to (N,)
        improved = (new_fitness < self.fitness) if self.min else (new_fitness > self.fitness)
        self.fitness[improved] = new_fitness[improved]
        return np.any(improved)
    
    def solve(self):
        """
        Execute the optimization using the Bat algorithm. Continuously update bats and check for stop criteria.
        Returns:
            The position and fitness of the best bat found.
        """
        i = 0
        state = {'iterations': i, 'current_fitness': np.min(self.fitness)}
        while not self.stop(state):
            self._update_bats()
            if self._update_fitness():
                state['current_fitness'] = np.min(self.fitness)
            state['iterations'] = i
            i += 1
        best_index = np.argmin(self.fitness) if self.min else np.argmax(self.fitness)
        return self.bat_pos[best_index], self.fitness[best_index]