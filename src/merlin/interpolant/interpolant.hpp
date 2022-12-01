// Copyright 2022 quocdang1998
#ifndef MERLIN_INTERPOLANT_INTERPOLANT_HPP_
#define MERLIN_INTERPOLANT_INTERPOLANT_HPP_

#include "merlin/array/nddata.hpp"  // merlin::NdData
#include "merlin/interpolant/grid.hpp"  // merlin::CartesianGrid
#include "merlin/vector.hpp"  // merlin::floatvec

namespace merlin {

class CartesianInterpolant {
  public:
    CartesianInterpolant(CartesianGrid & grid, array::NdData & value);
    ~CartesianInterpolant(void) = default;

    virtual float operator()(const floatvec & x) {return 0.0;}

  protected:
    CartesianGrid * grid_ = nullptr;
    array::NdData * value_ = nullptr;
};

class LagrangeInterpolant : public CartesianInterpolant {
  public:
    LagrangeInterpolant(CartesianGrid & grid, array::NdData & value);
    ~LagrangeInterpolant(void) = default;

  protected:
    /** @brief Interpolation coefficient vector.*/
    array::NdData coef_;
};

}  // namespace merlin

#endif  // MERLIN_INTERPOLANT_INTERPOLANT_HPP_
