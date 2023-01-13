// Copyright 2022 quocdang1998
#include "merlin/interpolant/cartesian_grid.hpp"

#include <cstdint>  // std::uintptr_t

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// CartesianGrid
// --------------------------------------------------------------------------------------------------------------------

void interpolant::CartesianGrid::copy_to_gpu(CartesianGrid * gpu_ptr, void * grid_vector_data_ptr) const {
    // initialize buffer to store data of the copy before cloning it to GPU
    interpolant::CartesianGrid copy_on_gpu;
    // shallow copy of grid vector
    copy_on_gpu.grid_vectors_.data() = reinterpret_cast<floatvec *>(grid_vector_data_ptr);
    copy_on_gpu.grid_vectors_.size() = this->ndim();
    // copy temporary object to GPU
    ::cudaMemcpy(gpu_ptr, &copy_on_gpu, sizeof(interpolant::CartesianGrid), cudaMemcpyHostToDevice);
    // copy data of each grid vector
    std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(grid_vector_data_ptr) + this->ndim()*sizeof(floatvec);
    for (int i = 0; i < this->ndim(); i++) {
        this->grid_vectors_[i].copy_to_gpu(&(copy_on_gpu.grid_vectors_[i]), reinterpret_cast<float *>(data_ptr));
        data_ptr += this->grid_vectors_[i].size() * sizeof(float);
    }
    // copy of grid shape
    // this->grid_shape_.copy_to_gpu(&(gpu_ptr->grid_shape_), reinterpret_cast<std::uint64_t *>(data_ptr));
    // nullify data pointer to avoid free data
    copy_on_gpu.grid_vectors_.data() = nullptr;
}

}  // namespace merlin
