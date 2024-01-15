// Copyright 2023 quocdang1998
#include "merlin/grid/regular_grid.hpp"

#include "merlin/array/array.hpp"  // merlin::array::Array
#include "merlin/logger.hpp"       // FAILURE

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

// Constructor number of points
grid::RegularGrid::RegularGrid(std::uint64_t num_points, std::uint64_t n_dim) : n_dim_(n_dim), num_points_(num_points) {
    this->grid_data_ = floatvec(smallest_power_of_2(num_points) * n_dim);
}

// Constructor from an array of point coordinates
grid::RegularGrid::RegularGrid(const array::Array & point_coordinates) {
    // check argument
    if (point_coordinates.ndim() != 2) {
        FAILURE(std::invalid_argument, "Expected array of ndim = 2.\n");
    }
    // allocate memory
    this->num_points_ = point_coordinates.shape()[0];
    this->n_dim_ = point_coordinates.shape()[1];
    this->grid_data_ = floatvec(smallest_power_of_2(this->num_points_) * this->n_dim_);
    // copy point coordinates
    intvec index(2);
    for (std::uint64_t i_point = 0; i_point < this->num_points_; i_point++) {
        index[0] = i_point;
        for (std::uint64_t i_dim = 0; i_dim < this->n_dim_; i_dim++) {
            index[1] = i_dim;
            this->grid_data_[i_point * this->n_dim_ + i_dim] = point_coordinates.get(index);
        }
    }
}

// Get element at a given flatten index
floatvec grid::RegularGrid::operator[](std::uint64_t index) const noexcept {
    floatvec point(this->n_dim_);
    this->get(index, point.data());
    return point;
}

}  // namespace merlin
