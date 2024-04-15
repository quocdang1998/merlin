// Copyright 2023 quocdang1998
#include "merlin/grid/regular_grid.hpp"

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// RegularGrid
// ---------------------------------------------------------------------------------------------------------------------

// Copy data to a pre-allocated memory
void * grid::RegularGrid::copy_to_gpu(grid::RegularGrid * gpu_ptr, void * grid_data_ptr,
                                      std::uintptr_t stream_ptr) const {
    // initialize buffer to store data of the copy before cloning it to GPU
    grid::RegularGrid copy_on_gpu;
    // copy grid ndim and size
    copy_on_gpu.ndim_ = this->ndim_;
    copy_on_gpu.num_points_ = this->num_points_;
    // assign pointer to GPU
    double * p_grid_data = reinterpret_cast<double *>(grid_data_ptr);
    copy_on_gpu.grid_data_.data() = p_grid_data;
    copy_on_gpu.grid_data_.size() = this->num_points_ * this->ndim_;
    // copy grid points to GPU
    ::cudaStream_t stream = reinterpret_cast<::cudaStream_t>(stream_ptr);
    ::cudaMemcpyAsync(grid_data_ptr, this->grid_data_.data(), copy_on_gpu.grid_data_.size() * sizeof(double),
                      ::cudaMemcpyHostToDevice, stream);
    // copy temporary object to GPU
    ::cudaMemcpyAsync(gpu_ptr, &copy_on_gpu, sizeof(grid::RegularGrid), ::cudaMemcpyHostToDevice, stream);
    // nullify pointer of temporary object to avoid de-allocate GPU pointer
    copy_on_gpu.grid_data_.data() = nullptr;
    return p_grid_data + copy_on_gpu.grid_data_.size();
}

}  // namespace merlin
