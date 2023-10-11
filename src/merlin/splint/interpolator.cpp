// Copyright 2022 quocdang1998
#include "merlin/splint/interpolator.hpp"

#include <omp.h>  // #pragma omp

#include "merlin/array/array.hpp"            // merlin::array::Array
#include "merlin/logger.hpp"                 // FAILURE
#include "merlin/splint/cartesian_grid.hpp"  // merlin::splint::CartesianGrid

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Interpolant
// ---------------------------------------------------------------------------------------------------------------------

// Construct from a CPU array
splint::Interpolant::Interpolant(const splint::CartesianGrid & grid, const array::Array & data,
                                 const Vector<splint::Method> & method) :
p_grid_(&grid), method_(method) {
    // check shape
    if (grid.shape() != data.shape()) {
        FAILURE(std::invalid_argument, "Grid and data have different shape.\n");
    }
    // copy data
    this->p_coeff_ = new array::Array(data);
}

// Destructor
splint::Interpolant::~Interpolant(void) {
    if (this->p_coeff_ != nullptr) {
        delete this->p_coeff_;
    }
}

}  // namespace merlin
