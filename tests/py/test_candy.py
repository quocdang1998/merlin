import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "../../../")))

import numpy as np

from merlin.array import Array, empty_array
from merlin.candy import Model, Gradient, create_grad_descent, Trainer, TrialPolicy
from merlin.candy.rand import Gaussian, Uniform
from merlin import Synchronizer

np_arr = np.array([[1.2, 2.4, 5.6], [3.7, 4.8, 5.3]], dtype=np.double)
data = Array(np_arr)

# calculate gradient
model = Model(data.shape, 2)
randomizer = [Gaussian(), Uniform(0.01)]
print(f"Randomizer: {randomizer}")
model.initialize(data, randomizer)
print("Gradient before trained :")
with Gradient(model.num_params) as g:
    g.calc(model, data)
    print(np.array(g.value()))

# dry run
optmz = create_grad_descent(0.22)
synch = Synchronizer("cpu")
tr = Trainer(1, synch)
key = "foo"
tr.set_model(key, model)
tr.set_optmz(key, optmz)
tr.set_data(key, data)
# tr.set_export_fname(key, "foo.txt")
policy = TrialPolicy(1, 9, 90)
dry_run_map = tr.dry_run([key], policy=policy, n_threads=2, metric="relsquare")
synch.synchronize()
if count < 1000:
    print(error)
    print(count)
    raise ValueError("Incompatible optimizer!")

# official run
tr.update_cpu(data, rep=100, threshold=0.001)
synch.synchronize()
print(f"Model after trained: {tr.model}")

# get gradient after trained
print("Gradient after trained :")
with Gradient(tr.model.num_params) as g:
    g.calc(tr.model, data)
    print(np.array(g.value()))

# get error after trained
err = tr.error_cpu(data, 4)
synch.synchronize()
print(err)
