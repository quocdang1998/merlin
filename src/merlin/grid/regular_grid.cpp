// Copyright 2023 quocdang1998
#include "merlin/grid/regular_grid.hpp"

#include <algorithm>  // std::copy, std::min
#include <cstring>    // std::memcpy
#include <sstream>    // std::ostringstream

#include "merlin/array/array.hpp"  // merlin::array::Array
#include "merlin/logger.hpp"       // merlin::Fatal, merlin::cuda_compile_error
#include "merlin/memory.hpp"       // merlin::memcpy_cpu_to_gpu

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------------------------------------------------

// Get the smallest power of 2 that is greater of equal a number
static inline std::uint64_t smallest_power_of_2(std::uint64_t n) {
    if (n == 0) {
        return 1;  // 2^0 = 1
    }
    std::uint64_t result = 1;
    while (result < n) {
        result <<= 1;  // Left shift to multiply by 2
    }
    return result;
}

// ---------------------------------------------------------------------------------------------------------------------
// RegularGrid
// ---------------------------------------------------------------------------------------------------------------------

// Reallocate memory for the grid
void grid::RegularGrid::realloc(std::uint64_t new_npoints) {
    DoubleVec new_grid_data(new_npoints * this->ndim_);
    std::uint64_t elem_to_cpy = std::min(new_grid_data.size(), this->grid_data_.size());
    std::memcpy(new_grid_data.data(), this->grid_data_.data(), elem_to_cpy * sizeof(double));
    this->grid_data_ = new_grid_data;
}

// Constructor number of points
grid::RegularGrid::RegularGrid(std::uint64_t ndim, std::uint64_t num_points) : ndim_(ndim), num_points_(num_points) {
    this->grid_data_ = DoubleVec(smallest_power_of_2(num_points) * ndim);
}

// Constructor from an array of point coordinates
grid::RegularGrid::RegularGrid(const array::Array & point_coordinates) {
    // check argument
    if (point_coordinates.ndim() != 2) {
        Fatal<std::invalid_argument>("Expected array of ndim = 2.\n");
    }
    // allocate memory
    this->num_points_ = point_coordinates.shape()[0];
    this->ndim_ = point_coordinates.shape()[1];
    this->grid_data_ = DoubleVec(smallest_power_of_2(this->num_points_) * this->ndim_);
    // copy point coordinates
    Index pt_index;
    for (std::uint64_t i_point = 0; i_point < this->num_points_; i_point++) {
        pt_index[0] = i_point;
        for (std::uint64_t i_dim = 0; i_dim < this->ndim_; i_dim++) {
            pt_index[1] = i_dim;
            this->grid_data_[i_point * this->ndim_ + i_dim] = point_coordinates.get(pt_index);
        }
    }
}

// Add a point to the grid
void grid::RegularGrid::push_back(const DoubleVec & new_point) noexcept {
    // argument checking
    if (new_point.size() != this->ndim_) {
        Fatal<std::invalid_argument>("Inappropriate ndim provided.\n");
    }
    // allocate larger memory if required
    if (this->num_points_ == smallest_power_of_2(this->num_points_)) {
        this->realloc(2 * this->num_points_);
    }
    // copy new point's coordinates
    std::memcpy(this->grid_data_.data() + this->num_points_ * this->ndim_, new_point.data(),
                this->ndim_ * sizeof(double));
    this->num_points_ += 1;
}

// Remove a point from the grid
Point grid::RegularGrid::pop_back(void) noexcept {
    // copy coordinates of the last point
    Point last_point;
    last_point.fill(0);
    double * last_point_coordinates = this->grid_data_.data() + (this->num_points_ - 1) * this->ndim_;
    std::copy(last_point_coordinates, last_point_coordinates + this->ndim_, last_point.data());
    // resize data if needed
    this->num_points_ -= 1;
    if (this->num_points_ == smallest_power_of_2(this->num_points_)) {
        this->realloc(this->num_points_);
    }
    return last_point;
}

// Copy data to a pre-allocated memory
void * grid::RegularGrid::copy_to_gpu(grid::RegularGrid * gpu_ptr, void * grid_data_ptr,
                                      std::uintptr_t stream_ptr) const {
    // initialize buffer to store data of the copy before cloning it to GPU
    grid::RegularGrid cloned_obj;
    // copy grid ndim and size
    cloned_obj.ndim_ = this->ndim_;
    cloned_obj.num_points_ = this->num_points_;
    // assign pointer to GPU
    double * p_grid_data = reinterpret_cast<double *>(grid_data_ptr);
    cloned_obj.grid_data_.data() = p_grid_data;
    cloned_obj.grid_data_.size() = this->num_points_ * this->ndim_;
    // copy grid points to GPU
    memcpy_cpu_to_gpu(grid_data_ptr, this->grid_data_.data(), cloned_obj.grid_data_.size() * sizeof(double),
                      stream_ptr);
    // copy temporary object to GPU
    memcpy_cpu_to_gpu(gpu_ptr, &cloned_obj, sizeof(grid::RegularGrid), stream_ptr);
    // nullify pointer of temporary object to avoid de-allocate GPU pointer
    cloned_obj.grid_data_.data() = nullptr;
    return p_grid_data + cloned_obj.grid_data_.size();
}

// String representation
std::string grid::RegularGrid::str(void) const {
    std::ostringstream os;
    os << "<RegularGrid(";
    DoubleVec point(this->ndim_);
    for (std::uint64_t i_point = 0; i_point < this->num_points_; i_point++) {
        if (i_point != 0) {
            os << " ";
        }
        this->get(i_point, point.data());
        os << point.str();
    }
    os << ")>";
    return os.str();
}

// Destructor
grid::RegularGrid::~RegularGrid(void) {}

}  // namespace merlin
