# Copyright 2022 quocdang1998

cdef extern from "merlin/device/gpu_query.hpp":
    int cpp_get_device_count "merlin::get_device_count" ()
    void cpp_print_device_limit "merlin::print_device_limit" (int device)
    bint cpp_test_gpu "merlin::test_gpu" (int device)
