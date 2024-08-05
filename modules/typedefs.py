from typing import Any
from numpy import ndarray, dtype, float32, float64, bool_

NDArrayFloat = ndarray[Any, dtype[float32 | float64]]
NDArrayBool = ndarray[Any, dtype[bool_]]
