import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "../../../")))

import numpy as np
from merlin.array import Array

x = np.array([0,1,2,3,4,5], dtype=np.double)
a = Array(x)
print(a)
b = Array(np.ones((2,3,4)))
print(b)
