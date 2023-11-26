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
    // shallow copy of grid nodes, grid shape and grid vectors
    double * grid_nodes_ptr = reinterpret_cast<double *>(grid_data_ptr);
    copy_on_gpu.grid_nodes_.data() = grid_nodes_ptr;
    copy_on_gpu.grid_nodes_.size() = this->num_nodes();
    std::uint64_t * grid_shape_ptr = reinterpret_cast<std::uint64_t *>(grid_nodes_ptr + this->num_nodes());
    copy_on_gpu.grid_shape_.data() = grid_shape_ptr;
    copy_on_gpu.grid_shape_.size() = this->ndim();
    Vector<double *> gpu_grid_vector = ptr_to_subsequence(grid_nodes_ptr, this->grid_shape_);
    double ** grid_vectors_ptr = reinterpret_cast<double **>(grid_shape_ptr + this->ndim());
    copy_on_gpu.grid_vectors_.data() = grid_vectors_ptr;
    copy_on_gpu.grid_vectors_.size() = this->ndim();
    copy_on_gpu.size_ = this->size_;
    // copy data of each vector
    ::cudaStream_t stream = reinterpret_cast<::cudaStream_t>(stream_ptr);
    ::cudaMemcpyAsync(grid_nodes_ptr, this->grid_nodes_.data(), this->num_nodes() * sizeof(double),
                      ::cudaMemcpyHostToDevice, stream);
    ::cudaMemcpyAsync(grid_shape_ptr, this->grid_shape_.data(), this->ndim() * sizeof(std::uint64_t),
                      ::cudaMemcpyHostToDevice, stream);
    ::cudaMemcpyAsync(grid_vectors_ptr, gpu_grid_vector.data(), this->ndim() * sizeof(double *),
                      ::cudaMemcpyHostToDevice, stream);
    // copy temporary object to GPU
    ::cudaMemcpyAsync(gpu_ptr, &copy_on_gpu, sizeof(grid::CartesianGrid), ::cudaMemcpyHostToDevice, stream);
    // nullify pointer of temporary object to avoid de-allocate GPU pointer
    copy_on_gpu.grid_nodes_.data() = nullptr;
    copy_on_gpu.grid_shape_.data() = nullptr;
    copy_on_gpu.grid_vectors_.data() = nullptr;
    return reinterpret_cast<void *>(grid_vectors_ptr + this->ndim());
}

}  // namespace merlin
