// Copyright 2022 quocdang1998
#include "merlin/array/array.hpp"

#include <cinttypes>   // PRIu64
#include <functional>  // std::bind, std::placeholders

#include "merlin/array/operation.hpp"  // merlin::array::copy
#include "merlin/array/parcel.hpp"     // merlin::array::Parcel
#include "merlin/cuda/device.hpp"      // merlin::cuda::Device
#include "merlin/env.hpp"              // merlin::Environment
#include "merlin/logger.hpp"           // merlin::Fatal, merlin::Warning

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Memory lock (allocated array always stays in the RAM)
// ---------------------------------------------------------------------------------------------------------------------

// Allocate non pageable memory
double * array::allocate_memory(std::uint64_t size) {
    double * result = nullptr;
    ::cudaError_t err_ = ::cudaMallocHost(&result, sizeof(double) * size);
    if (err_ != 0) {
        Fatal<cuda_runtime_error>("Allocate non-pageable memory failed with message \"%s\".\n",
                                  ::cudaGetErrorString(err_));
    }
    return result;
}

// Pin memory to RAM
void array::cuda_pin_memory(double * ptr, std::uint64_t mem_size) {
    ::cudaError_t err_ = ::cudaHostRegister(ptr, mem_size, cudaHostRegisterDefault);
    if (err_ != 0) {
        Warning("Pin pageable memory failed with message \"") << ::cudaGetErrorString(err_) << "\".\n";
    }
}

// Unpin memory
void array::cuda_unpin_memory(double * ptr) {
    ::cudaError_t err_ = ::cudaHostUnregister(ptr);
    if (err_ != 0) {
        Warning("Unpin paged memory failed with message \"") << ::cudaGetErrorString(err_) << "\".\n";
    }
}

// Free non pageable memory
void array::free_memory(double * ptr) {
    ::cudaError_t err_ = ::cudaFreeHost(ptr);
    if (err_ != 0) {
        Fatal<cuda_runtime_error>("Free non-pageable memory failed with message \"%s\".\n", ::cudaGetErrorName(err_));
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Array
// ---------------------------------------------------------------------------------------------------------------------

// Copy data from GPU array
void array::Array::clone_data_from_gpu(const array::Parcel & src, const cuda::Stream & stream) {
    // check GPU of stream
    if (src.device() != stream.get_gpu()) {
        Fatal<cuda_runtime_error>("Cannot copy from GPU array (%d) with stream pointing to another GPU (%d).\n",
                                  src.device(), stream.get_gpu());
    }
    // cast stream
    ::cudaStream_t copy_stream = reinterpret_cast<::cudaStream_t>(stream.get_stream_ptr());
    // create copy function
    auto copy_func = std::bind(::cudaMemcpyAsync, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                               ::cudaMemcpyDeviceToHost, copy_stream);
    // copy data to GPU
    std::uintptr_t current_ctx = src.device().push_context();
    array::copy(dynamic_cast<array::NdData *>(this), dynamic_cast<const array::NdData *>(&src), copy_func);
    cuda::Device::pop_context(current_ctx);
}

}  // namespace merlin
