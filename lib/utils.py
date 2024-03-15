# %%

from typing import List, Dict, Tuple, Callable, Sequence, Any
import numpy as np

def get_initializer(name: str) -> Callable:
    return {
        "normal": lambda pop_size, dim: np.random.normal((pop_size, dim)),
        "zeros": lambda pop_size, dim: np.zeros((pop_size, dim)),
    }[name]