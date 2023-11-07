import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "../../../")))

import merlin.cuda

num_gpu = merlin.cuda.Device.get_num_gpu()

for i in range(num_gpu):
    gpu = merlin.cuda.Device(id=i)
    print(gpu)
    gpu.print_specification()
    gpu.test_gpu()

