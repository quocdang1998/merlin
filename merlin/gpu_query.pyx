# Copyright 2022 quocdang1998

from gpu_query cimport *

def get_device_count():
    return cpp_get_device_count()

def print_device_limit(int device = -1):
    cpp_print_device_limit(device)

def test_gpu(int device = -1):
    cpp_test_gpu(device)
