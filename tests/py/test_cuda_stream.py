import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "../../../")))

import merlin.device

num_gpu = merlin.device.Device.get_num_gpu()

for i in range(num_gpu):
    gpu = merlin.device.Device(id=i)
    gpu.print_specification()
    gpu.test_gpu()

