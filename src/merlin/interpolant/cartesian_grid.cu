// Copyright 2022 quocdang1998
#include "merlin/interpolant/cartesian_grid.hpp"

#include <cstdint>  // std::uintptr_t

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// CartesianGrid
// --------------------------------------------------------------------------------------------------------------------

void * interpolant::CartesianGrid::copy_to_gpu(interpolant::CartesianGrid * gpu_ptr, void * grid_vector_data_ptr,
                                               std::uintptr_t stream_ptr) const {
    // initialize buffer to store data of the copy before cloning it to GPU
    interpolant::CartesianGrid copy_on_gpu;
    // shallow copy of grid vector
    copy_on_gpu.grid_vectors_.data() = reinterpret_cast<floatvec *>(grid_vector_data_ptr);
    copy_on_gpu.grid_vectors_.size() = this->ndim();
    // copy data of each grid vector
    std::uintptr_t dptr = reinterpret_cast<std::uintptr_t>(grid_vector_data_ptr) + this->ndim()*sizeof(floatvec);
    void * data_ptr = reinterpret_cast<void *>(dptr);
    for (std::uint64_t i_dim = 0; i_dim < this->ndim(); i_dim++) {
        data_ptr = this->grid_vectors_[i_dim].copy_to_gpu(&(copy_on_gpu.grid_vectors_[i_dim]), data_ptr, stream_ptr);
    }
    // copy temporary object to GPU
    ::cudaMemcpyAsync(gpu_ptr, &copy_on_gpu, sizeof(interpolant::CartesianGrid), ::cudaMemcpyHostToDevice,
                      reinterpret_cast<::cudaStream_t>(stream_ptr));
    // nullify data pointer to avoid free data
    copy_on_gpu.grid_vectors_.data() = nullptr;
    return data_ptr;
}

}  // namespace merlin
