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
grid::RegularGrid::RegularGrid(std::uint64_t num_points, std::uint64_t ndim) : ndim_(ndim), num_points_(num_points) {
    this->grid_data_ = DoubleVec(smallest_power_of_2(num_points) * ndim);
}

// Constructor from an array of point coordinates
grid::RegularGrid::RegularGrid(const array::Array & point_coordinates) {
    // check argument
    if (point_coordinates.ndim() != 2) {
        FAILURE(std::invalid_argument, "Expected array of ndim = 2.\n");
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

// Destructor
grid::RegularGrid::~RegularGrid(void) {}

}  // namespace merlin
