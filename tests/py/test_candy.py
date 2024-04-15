import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "../../../")))

import numpy as np

from merlin.array import Array, empty_array
from merlin.candy import Model, Gradient, create_grad_descent, Trainer

np_arr = np.array([[1.2, 2.4, 5.6], [3.7, 4.8, 5.3]], dtype=np.double)
data = Array(np_arr)

model = Model(data.shape, 2)
model.initialize(data)
with Gradient(model.num_params) as g:
    g.calc(model, data)
    print(np.array(g.value()))

optmz = create_grad_descent(0.1)
tr = Trainer(model, optmz)
tr.update_cpu(data, rep=100, threshold=0.01)
tr.synchronize()
