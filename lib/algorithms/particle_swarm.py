import numpy as np
from typing import Callable
from lib.utils import define_stop_criterion


class PSO:
    """A Particle Swarm Optimization Algorithm Impelmentation.
    """
    def __init__(self, config: object) -> None:
        """Initializes the PSO state variables and params.
        Args:
            config (object): Custom data structure to hold optimization algorithm parameters
        """
        # Optimization Problem
        self.D = config.D   # dimension search space
        self.f = config.funct   # Optimization Function
        
        # Stop Criterion
        self.stop = define_stop_criterion(config)
        
        # agent specific data (a)
        self.a_pos = np.random.uniform(config.lb, config.ub, size=(config.population, config.D))
        self.a_velocity = np.random.uniform(0, config.v_init, size=(config.population, config.D))
        self.a_pos_star = self.a_pos.copy() # initial personal best

        self.a_fit_star = self._get_fitness() # Personal best scores
        
        # Global best (position)
        self.g_star = self.a_pos[np.argmin(self.a_fit_star)]
        
        # Add'l params
        self.alpha = config.alpha
        self.beta = config.beta
        self.v_max = config.v_max
        self.ub = config.ub
        self.lb = config.lb
    
    def _update_velocity(self):
        """Updates the velocity of each particle.
        """
        eps1 = np.random.uniform(0, 1, size=(self.a_pos.shape[0], self.D))
        eps2 = np.random.uniform(0, 1, size=(self.a_pos.shape[0], self.D))
        
        alpha_step = self.alpha * eps1 * (self.g_star - self.a_pos)
        beta_step = self.beta * eps2 * (self.a_pos_star - self.a_pos)
        self.a_velocity += alpha_step + beta_step 
        
        np.clip(self.a_velocity, -self.v_max, self.v_max, out=self.a_velocity)
    
    def _update_pos(self):
        """Updates the position of each partice
        """
        self.a_pos += self.a_velocity
        self.a_pos = np.clip(self.a_pos, self.lb, self.ub)
    
    def _update_a_pos_star(self):
        """Updates the best local position of each particle
        """
        curr_fit = self._get_fitness() # shape(N,)
        better_fit_indices = curr_fit < self.a_fit_star

        self.a_pos_star[better_fit_indices] = self.a_pos[better_fit_indices]
        self.a_fit_star[better_fit_indices] = curr_fit[better_fit_indices]

    def _update_g_star(self):
        """Updates the global optima
        """
        fit_star_idx = np.argmin(self.a_fit_star)
        self.g_star = self.a_pos_star[fit_star_idx]
    
    def _get_fitness(self) -> np.ndarray:
        """Calculates the fitness of the current generation (each particle)
        Returns:
            np.ndarray: fitness scores for each particle
        """
        curr_fit = np.apply_along_axis(self.f, 1, self.a_pos) # shape(N, 1)
        # Flatten to ensure 1D
        curr_fit = curr_fit.ravel() # shape(N,)
        return curr_fit
    
    def solve(self) -> tuple:
        """Runs each generation of the algorithm.
        Returns:
            tuple: the best agents position, its fitness
        """
        i = 0
        state = {
            'iterations': i,
            'current_fitness': min(self.a_fit_star),
            # Include more statements for stop criterions here
        }
        
        # Loop until stop criteria are met
        while not self.stop(state):
            self._update_velocity()
            self._update_pos()
            self._update_a_pos_star()
            self._update_g_star()
            
            # Update state with new values
            state['iterations'] = i
            state['current_fitness'] = min(self.a_fit_star)

            i += 1

        best_fitness = min(self.a_fit_star)
        best_position = self.g_star
        return best_position, best_fitness
        