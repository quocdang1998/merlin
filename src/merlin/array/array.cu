// Copyright 2022 quocdang1998
#include "merlin/array/array.hpp"

#include <functional>  // std::bind, std::placeholders

#include "merlin/array/copy.hpp"  // merlin::array::array_copy
#include "merlin/array/parcel.hpp"  // merlin::array::Parcel
#include "merlin/cuda/device.hpp"  // merlin::cuda::Device
#include "merlin/logger.hpp"  // FAILURE

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// Array
// --------------------------------------------------------------------------------------------------------------------

// Copy data from GPU array
void array::Array::clone_data_from_gpu(const array::Parcel & src, const cuda::Stream & stream) {
    // save current gpu
    cuda::Device current_gpu = cuda::Device::get_current_gpu();
    // check GPU of stream
    if (src.device() != stream.get_gpu()) {
        FAILURE(cuda_runtime_error, "Cannot copy from GPU array (%d) with stream pointing to another GPU (%d).\n",
                src.device(), stream.get_gpu());
    }
    // cast stream
    ::cudaStream_t copy_stream = reinterpret_cast<::cudaStream_t>(stream.get_stream_ptr());
    // create copy function
    auto copy_func = std::bind(::cudaMemcpyAsync, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                               ::cudaMemcpyDeviceToHost, copy_stream);
    // copy data to GPU
    src.device().set_as_current();
    array::array_copy(dynamic_cast<array::NdData *>(this), dynamic_cast<const array::NdData *>(&src), copy_func);
    current_gpu.set_as_current();
}

}  // namespace merlin
