"""With these lines, within any files in the lib folder, you can import `lib`, for example:
```
import lib
import lib.algorithms
from lib.benchmarks import Ackley
```
"""

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from .config import Config
from .types import (MetricsType,)

