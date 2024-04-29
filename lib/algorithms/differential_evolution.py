import numpy as np
from typing import Callable
from lib.utils import define_stop_criterion

class DE:
    """A Differentia Evolution Algorithm Impelmentation.
    """
    def __init__(self, config: object) -> None:
        """Initializes the Differential Evolution state variables and params.
        Args:
            config (object): Custom data structure to hold optimization algorithm parameters
        """
        self.agents = np.random.uniform(config.lb, config.ub, size=(config.population, config.D))
        self.D = config.D
        self.F = config.differential_weight
        self.C = config.crossover_param
        self.f = config.funct
        self.stop = define_stop_criterion(config)
        
        self.crossover: Callable = self._define_crossover(config.crossover_scheme)
        self.lb = config.lb
        self.ub = config.ub
    
    def _define_crossover(self, name : str) -> Callable:
        """Defines the crossover function.
        Args:
            name (str): name of the crossover type
        Returns:
            Callable: The function used for crossover
        """
        return {
            "binomial": self._binomial_crossover,
            "exponential": self._exponential_crossover,
        }[name]
    
    def _binomial_crossover(self, donor_vec: np.ndarray) -> np.ndarray:
        """Performs the binomial crossover.
        Args:
            donor_vec (np.ndarray): The donor vector from mutation.
        Returns:
            np.ndarray: The trial vector as a potential update
        """
        mask = np.random.rand(self.agents.shape[0], self.D) < self.C
        j_rand = np.random.randint(0, self.D, size=(self.agents.shape[0], 1))
        np.put_along_axis(mask, j_rand, True, axis=1)
        trial_vecs = np.where(mask, donor_vec, self.agents)
        trial_vecs = np.clip(trial_vecs, self.lb, self.ub)
        return trial_vecs
        
    def _exponential_crossover(self, donor_vec: np.ndarray) -> np.ndarray:
        """Performs the exponential crossover.
        Args:
            donor_vec (np.ndarray): The donor vector from mutation.
        Returns:
            np.ndarray: The trial vector as a potential update
        """
        n_agents, D = self.agents.shape
        trial_vecs = np.copy(self.agents)
        
        for i in range(n_agents):
            k = np.random.randint(0, D) # starting index
            L = np.random.randint(1, D + 1) # len of segment copied
            indices = (k + np.arange(L)) % D
            # Crossover
            trial_vecs[i, indices] = donor_vec[i, indices]
        trial_vecs = np.clip(trial_vecs, self.lb, self.ub)
        return trial_vecs
    
    def _mutation(self) -> np.ndarray:
        """Performs mutation by selecting 3 distinct random positions.
        Returns:
            np.ndarray: donor vector for crossover
        """
        donor_vecs = np.zeros_like(self.agents)
        
        for i in range(len(self.agents)):
            idx_choices = [idx for idx in range(self.agents.shape[0]) if idx != i]
            x_p, x_q, x_r = np.random.choice(idx_choices, 3, replace=False)
        
            donor_vecs[i] = self.agents[x_p] + self.F * (self.agents[x_q] - self.agents[x_r])
        donor_vecs = np.clip(donor_vecs, self.lb, self.ub)
        return donor_vecs

    def _selection(self, trial_vecs: np.ndarray) -> tuple:
        """Checks if fittness of the trial vector from crossover is
        better than the current one and updates if that is the case.
        Args:
            trial_vecs (np.ndarray): The trial vector as a potential update
        Returns:
            tuple: the updatet agents, their fittness score
        """
        trial_fitness = np.apply_along_axis(self.f, 1, trial_vecs)
        current_fitness = np.apply_along_axis(self.f, 1, self.agents)
        
        # Flatten the arrays to ensure they are 1D
        trial_fitness = trial_fitness.ravel()
        current_fitness = current_fitness.ravel()

        better_mask = trial_fitness <= current_fitness
        self.agents = np.where(better_mask[:, None], trial_vecs, self.agents)

        fit_score = np.apply_along_axis(self.f, 1, self.agents)
        return self.agents, fit_score
    
    def solve(self):
        """Runs each generation of the algorithm.
        Returns:
            tule: the best agents position, its fitness
        """
        i = 0
        best_agent = None, np.inf
        state = {'iterations': 0, 'current_fitness': np.inf}
        while not self.stop(state):
            donor_vecs = self._mutation()
            trial_vecs = self.crossover(donor_vecs)
            self.agents, fit_scores = self._selection(trial_vecs)
            best_idx = np.argmin(fit_scores)

            if best_agent[1] > fit_scores[best_idx]:
                best_agent = self.agents[best_idx], fit_scores[best_idx]
            state['iterations'] = i
            state['current_fitness'] = best_agent[1]
            i += 1
        return best_agent
        