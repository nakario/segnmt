from typing import Union
import numpy
try:
    import cupy
except ImportError:
    cupy = numpy


ndarray = Union[numpy.ndarray, cupy.ndarray]
