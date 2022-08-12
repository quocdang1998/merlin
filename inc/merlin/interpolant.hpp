// Copyright 2022 quocdang1998
#ifndef MERLIN_INTERPOLANT_HPP_
#define MERLIN_INTERPOLANT_HPP_

#include <vector>

#include "merlin/tensor.hpp"
#include "merlin/grid.hpp"

namespace merlin {

class CartesianInterpolant {
  public:
    CartesianInterpolant(CartesianGrid & grid, Tensor & value);
    ~CartesianInterpolant(void) = default;

    virtual float operator() (const std::vector<float> & x);

  protected:
    CartesianGrid * grid_;
    Tensor * value_;
};

class LagrangeInterpolant : public CartesianInterpolant {
  public:
    LagrangeInterpolant(CartesianGrid & grid, Tensor & value);
    ~LagrangeInterpolant(void) = default;

  protected:
    /** @brief Interpolation coefficient vector.*/
    Tensor coef_;
};

}  // namespace merlin

#endif  // MERLIN_INTERPOLANT_HPP_
