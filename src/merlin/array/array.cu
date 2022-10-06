// Copyright 2022 quocdang1998
#include "merlin/array/array.hpp"

#include <functional>  // std::bind, std::placeholders

#include "merlin/array/parcel.hpp"  // merlin::Parcel
#include "merlin/logger.hpp"  // FAILURE
#include "merlin/array/utils.hpp"  // merlin::array_copy

namespace merlin {

void Array::sync_from_gpu(const Parcel & gpu_array, std::uintptr_t stream) {
    // check device
    int check_result = gpu_array.check_device();
    if (check_result != 0) {
        FAILURE(cuda_runtime_error, "Current GPU is not corresponding (expected ID %d, got ID %d).\n",
                gpu_array.device_id(), gpu_array.device_id() - check_result);
    }
    // cast stream
    cudaStream_t copy_stream = reinterpret_cast<cudaStream_t>(stream);
    // create copy function
    auto copy_func = std::bind(cudaMemcpyAsync, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                               cudaMemcpyDeviceToHost, copy_stream);
    // copy data to GPU
    array_copy(dynamic_cast<NdData *>(this), dynamic_cast<const NdData *>(&gpu_array), copy_func);
}

}  // namespace merlin
