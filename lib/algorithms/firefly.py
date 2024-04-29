import numpy as np
from typing import Callable
from lib.utils import define_stop_criterion

class FireFly:
    
    def __init__(self, config: object) -> None:
        """
        Initialize the FireFly algorithm with a configuration object.
        Parameters:
            config: An object containing all necessary settings, including:
                - D (dimension of the problem)
                - funct (objective function to optimize)
                - min (boolean indicating minimization or not)
                - max_iter_no_improv (maximum iterations without improvement for stopping)
                - max_i (maximum iterations)
                - lb, ub (lower and upper bounds for the search space)
                - population (number of fireflies)
                - beta0, gamma, alpha (algorithm parameters for movement and intensity update)
        """
        # Optimization Problem
        self.D = config.D
        self.f = config.funct
        
        # Minimization or Maximization Problem
        self.min = True if hasattr(config, "min") else False
        
        # Stop Criterion
        self.stop: Callable = define_stop_criterion(config)
        
        # firefly
        self.fireflies = np.random.uniform(config.lb, config.ub, size=(config.population, config.D))
        self.intensity = self._get_fitness()
        
        # self.alpha_decay_factor = config.alpha_decay_factor # to adjust alpha after every iteration
        self.beta0 = config.beta0
        self.gamma = config.gamma
        self.alpha = config.alpha
        self.ub = config.ub
        self.lb = config.lb

    def _get_fitness(self):
        """
        Evaluate the fitness of all fireflies based on the objective function.
        Returns:
            An array of fitness values for each firefly.
        """
        objective = np.apply_along_axis( self.f, 1, self.fireflies)
        return  (objective) if self.min else (-objective)
    
    def _move_fireflies(self):
        """
        Move fireflies towards brighter ones and apply random perturbations.
        Updates the positions of fireflies based on attractiveness and distance to other fireflies.
        """
        for i in range(len(self.fireflies)):
            # Calculate the distance matrix between firefly i and all other fireflies
            distances = np.linalg.norm(self.fireflies[i] - self.fireflies, axis=1)
            betas = self.beta0 / (1 + self.gamma * distances ** 2)
            
            # Update positions
            for j in range(len(self.fireflies)):
                if self.intensity[j] > self.intensity[i]:
                    self.fireflies[i] += betas[j] * (self.fireflies[j] - self.fireflies[i])
            
            # Add random perturbation and clip
            self.fireflies[i] += self.alpha * (np.random.rand(self.D) - 0.5)
            self.fireflies[i] = np.clip(self.fireflies[i], self.lb, self.ub)
            
    def _update_intensity(self):
        """
        Update the intensity (fitness) of each firefly if improvements are found.
        Returns:
            True if any firefly's intensity was improved, False otherwise.
        """
        new_intensity = self._get_fitness()
        improved = new_intensity < self.intensity
        self.intensity[improved] = new_intensity[improved]
        return np.any(improved)
    
    def solve(self):
        """
        Execute the Firefly Algorithm to find the best solution.
        Iteratively updates firefly positions and intensities until stopping criteria are met.
        Returns:
            The position and intensity of the best solution found.
        """
        i = 0
        state = {'iterations': i, 'current_fitness': np.min(self.intensity)}
        while not self.stop(state):
            self._move_fireflies()
            if self._update_intensity():
                state['current_fitness'] = np.min(self.intensity)
            state['iterations'] = i
            i += 1

        best_index = np.argmin(self.intensity)
        return self.fireflies[best_index], self.intensity[best_index]