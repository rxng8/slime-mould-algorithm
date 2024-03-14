
from typing import List, Dict, Tuple, Callable
import numpy as np

MetricsType = Dict[str, np.ndarray]
"""This is a dictionary of metrics, or what can be graphable. For example:
{
  "loss": (T,)
}
"""