// Copyright 2022 quocdang1998
#ifndef MERLIN_DEVICE_GPU_QUERY_HPP_
#define MERLIN_DEVICE_GPU_QUERY_HPP_

#include "merlin/device/decorator.hpp"  // __cuhostdev__
#include "merlin/exports.hpp"  // MERLIN_EXPORTS

namespace merlin::device {

/** @brief Get total number of GPU.*/
MERLIN_EXPORTS int get_device_count(void);

#ifdef __NVCC__
/** @brief Get ID of current active device.*/
__cuhostdev__ inline int get_current_device(void) {
    int current_device;
    cudaGetDevice(&current_device);
    return current_device;
}
#endif  // __NVCC__

/** @brief Print GPU specifications.
 *  @details Print GPU specifications (number of threads, total global memory, max shared memory) and API limitation (max
 *  thread per block, max block per grid).
 *  @param device ID of device (an integer ranging from 0 to the number of device). A value of `-1` will print details of
 *  all GPU.
 */
MERLIN_EXPORTS void print_device_limit(int device = -1);

/** @brief Perform an addition of two integers on GPU.
 *  @details This function tests if the installed CUDA is compatible with the GPU.
 *  @return ``true`` if all tests on all GPU pass.
 */
MERLIN_EXPORTS bool test_gpu(int device = -1);

}  // namespace merlin::device

#endif  // MERLIN_DEVICE_GPU_QUERY_HPP_
