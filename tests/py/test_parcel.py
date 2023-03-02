import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "../../../")))

import numpy as np
from merlin.array import Array, Parcel

x = np.array([[0,1],[2,3],[4,5]], dtype=np.double)
a = Array(array=x)
print(f"Original array: {a}")
p = Parcel(shape=a.shape)
print(f"Before transfering to GPU: {p}")
p.transfer_data_to_gpu(a)
print(f"After transfered to GPU: {p}")
