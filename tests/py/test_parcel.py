import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "../../../")))

import numpy as np
from merlin.array import Array, empty_parcel
from merlin.cuda import Stream

x = np.array([[0,1],[2,3],[4,5]], dtype=np.double)
a = Array(x)
print(f"Original array: {a}")
p = empty_parcel(a.shape)
print(f"Before transfering to GPU: {p}")
s = Stream(setting="nonblocking")
p.transfer_data_to_gpu(a, s)
s.synchronize()
print(f"After transfered to GPU: {p}")
