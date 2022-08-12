// Copyright 2022 quocdang1998
#include "merlin/interpolant.hpp"

#include <thread>

#include "merlin/logger.hpp"

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// CartesianInterpolant
// --------------------------------------------------------------------------------------------------------------------

CartesianInterpolant::CartesianInterpolant(CartesianGrid & grid, Tensor & value) : grid_(&grid), value_(&value) {
    // check number of points of grid and value vector
    if (this->grid_->npoint() != this->value_->size()) {
        FAILURE("Size of Grid (%d) and sizeof value tensor (%d) are not the same.",
                this->grid_->npoint(), this->value_->size());
    }
    // check number of dimension of grid and value vector
    if (this->grid_->ndim() != this->value_->ndim()) {
        FAILURE("Inconsistent ndim of Grid (%d) and ndim of value tensor (%d).",
                this->grid_->ndim(), this->value_->ndim());
    }
}

float CartesianInterpolant::operator() (const std::vector<float> & x) {
    return 0.0;
}

// --------------------------------------------------------------------------------------------------------------------
// LagrangeInterpolant
// --------------------------------------------------------------------------------------------------------------------

LagrangeInterpolant::LagrangeInterpolant(CartesianGrid & grid, Tensor & value) : CartesianInterpolant(grid, value) {
    // copy value to coef_ tensor
    this->coef_ = value;
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
        MESSAGE("%f", this->coef_[it.index()]);
    }
}

}  // namespace merlin
