import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "../../../")))

import numpy as np

from merlin.array import Array
from merlin.grid import CartesianGrid
from merlin.splint import Method, Interpolator

def foo(x):
    return (2.0*x[0] + x[2])*x[2] + 3.0*x[1];

grid = CartesianGrid(grid_vectors=[[0.1, 0.2, 0.3], [1.0, 2.0, 3.0, 4.0], [0.0, 0.25, 0.5]])
data = []
for i in range(grid.size):
    data.append(foo(grid.get(i)))
np_buffer = np.array(data)
np_buffer = np_buffer.reshape(grid.shape)
value=Array(buffer=np_buffer, copy=False)
methods = [Method.Newton] * 3
interp = Interpolator(grid=grid, values=value, method=methods)
interp.build_coefficients()

points = np.array([[0.1, 1.2, 0.3], [0.24, 2.6, 0.12]], dtype=np.float64)
points_ml = Array(buffer=points, copy=False)
result = np.array(interp.evaluate(points=points_ml, n_threads=5), copy=False)
interp.synchronize()
print(result)

