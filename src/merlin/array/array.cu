// Copyright 2022 quocdang1998
#include "merlin/array/array.hpp"

#include <functional>  // std::bind, std::placeholders

#include "merlin/array/copy.hpp"  // merlin::array::array_copy
#include "merlin/array/parcel.hpp"  // merlin::array::Parcel
#include "merlin/cuda/gpu_query.hpp"  // merlin::cuda::Device
#include "merlin/logger.hpp"  // FAILURE

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// Array
// --------------------------------------------------------------------------------------------------------------------

// Copy data from GPU array
void array::Array::sync_from_gpu(const array::Parcel & gpu_array, std::uintptr_t stream) {
    // check device
    cuda::Device current_gpu = cuda::Device::get_current_gpu();
    if (current_gpu != gpu_array.device()) {
        FAILURE(cuda_runtime_error, "Current GPU is not corresponding (expected ID %d, got ID %d).\n",
                gpu_array.device().id(), current_gpu.id());
    }
    // cast stream
    cudaStream_t copy_stream = reinterpret_cast<cudaStream_t>(stream);
    // create copy function
    auto copy_func = std::bind(cudaMemcpyAsync, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                               cudaMemcpyDeviceToHost, copy_stream);
    // copy data to GPU
    array::array_copy(dynamic_cast<array::NdData *>(this), dynamic_cast<const array::NdData *>(&gpu_array), copy_func);
}

}  // namespace merlin
