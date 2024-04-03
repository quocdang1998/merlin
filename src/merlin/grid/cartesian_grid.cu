// Copyright 2022 quocdang1998
#include "merlin/grid/cartesian_grid.hpp"

#include "merlin/utils.hpp"  // merlin::ptr_to_subsequence

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// CartesianGrid
// ---------------------------------------------------------------------------------------------------------------------

void * grid::CartesianGrid::copy_to_gpu(grid::CartesianGrid * gpu_ptr, void * grid_data_ptr,
                                        std::uintptr_t stream_ptr) const {
    // initialize buffer to store data of the copy before cloning it to GPU
    grid::CartesianGrid copy_on_gpu;
    // copy grid shape, ndim and size
    copy_on_gpu.grid_shape_ = this->grid_shape_;
    copy_on_gpu.ndim_ = this->ndim_;
    copy_on_gpu.size_ = this->size_;
    // assign to grid data pointer
    double * p_grid_nodes = reinterpret_cast<double *>(grid_data_ptr);
    copy_on_gpu.grid_nodes_.data() = p_grid_nodes;
    copy_on_gpu.grid_nodes_.size() = this->num_nodes();
    copy_on_gpu.grid_vectors_.fill(nullptr);
    ptr_to_subsequence(p_grid_nodes, this->grid_shape_.data(), this->ndim_, copy_on_gpu.grid_vectors_.data());
    // copy node vector
    ::cudaStream_t stream = reinterpret_cast<::cudaStream_t>(stream_ptr);
    ::cudaMemcpyAsync(p_grid_nodes, this->grid_nodes_.data(), this->num_nodes() * sizeof(double),
                      ::cudaMemcpyHostToDevice, stream);
    // copy temporary object to GPU
    ::cudaMemcpyAsync(gpu_ptr, &copy_on_gpu, sizeof(grid::CartesianGrid), ::cudaMemcpyHostToDevice, stream);
    // nullify pointer of temporary object to avoid de-allocate GPU pointer
    copy_on_gpu.grid_nodes_.data() = nullptr;
    return p_grid_nodes + this->num_nodes();
}

}  // namespace merlin
