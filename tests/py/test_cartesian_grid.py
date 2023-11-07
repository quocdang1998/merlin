import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(__file__, "../../../")))

from merlin.array import Array, Parcel
from merlin.cuda import Stream, StreamSetting
from merlin.splint import CartesianGrid, Interpolator, Method

# test interpolation on CPU

cart_grid = CartesianGrid(grid_vectors=[[1.2,2.3,3.4],[5.4,1.9,6.5,8.9,7.1],[4.2,3.6,9.6,3.5]])
print(cart_grid)

methods = [Method.Newton] * 3
print(methods)

def foo(x, y, z):
    return 2*x*(y-3*z)

values_list = [foo(*cart_grid.get_pt_cindex(i)) for i in range(cart_grid.size)]
values_array = np.array(values_list, dtype=np.float64).reshape(cart_grid.shape)
values = Array(array=values_array, copy=True)

interpolator = Interpolator(grid=cart_grid, values=values, methods=methods, n_threads=32)
print(values)

# test interpolation on GPU
'''
stream = Stream(setting=StreamSetting.NonBlocking)

values_gpu = Parcel(shape=values.shape)
values_gpu.transfer_data_to_gpu(Array(array=values_array, copy=False), stream)
interpolator_gpu = Interpolator(grid=cart_grid, values=values_gpu, methods=methods, stream=stream, n_threads=32)
stream.synchronize()
print(values_gpu)
'''
