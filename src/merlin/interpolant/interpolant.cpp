// Copyright 2022 quocdang1998
#include "merlin/interpolant/interpolant.hpp"

#include <thread>

#include "merlin/logger.hpp"  // FAILURE
#include "merlin/vector.hpp"  // merlin::intvec

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// CartesianInterpolant
// --------------------------------------------------------------------------------------------------------------------

CartesianInterpolant::CartesianInterpolant(CartesianGrid & grid, array::NdData & value) :
grid_(&grid), value_(&value) {
    // check number of points of grid and value vector
    if (grid.ndim() != value.ndim()) {
        FAILURE(std::invalid_argument, "Ndim of Grid (%d) and value tensor (%d) are inconsistent.\n",
                grid.ndim(), value.ndim());
    }
    // check number of dimension of grid and value vector
    intvec grid_shape = grid.grid_shape();
    for (int i = 0; i < grid.ndim(); i++) {
        if (grid_shape[i] < value.shape()[i]) {
            FAILURE(std::invalid_argument, "Expected shape Grid (%d) less or equal value tensor (%d) at index %d.\n",
                    grid_shape[i], value.shape()[i], i);
        }
    }
}


// --------------------------------------------------------------------------------------------------------------------
// LagrangeInterpolant
// --------------------------------------------------------------------------------------------------------------------

LagrangeInterpolant::LagrangeInterpolant(CartesianGrid & grid, array::NdData & value) :
CartesianInterpolant(grid, value) {
    // copy value to coef_ tensor
    this->coef_ = value;
    /*
    // loop on each point in value
    for (Tensor::iterator it = this->coef_.begin(); it != this->coef_.end(); it++) {
        // calculate f(x_i) / prod((x-i-x_j) for j != i)
        float weight_ = 1.0;
        std::vector<unsigned int> & index_ = it.index();
        for (int dim_ = 0; dim_ < index_.size(); dim_++) {
            std::vector<float> & dim_values = this->grid_->grid_vectors()[dim_];
            for (int j = 0; j < dim_values.size(); j++) {
                if (j == index_[dim_]) {
                    continue;  // skip if j == i
                }
                weight_ *= (dim_values[index_[dim_]] - dim_values[j]);
            }
        }
        // multiply weight to coef
        float & coef_ = this->coef_[it.index()];
        coef_ = static_cast<float>(coef_) / static_cast<float>(weight_);
    }
    */
}

}  // namespace merlin
