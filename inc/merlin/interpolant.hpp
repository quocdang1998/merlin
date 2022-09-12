// Copyright 2022 quocdang1998
#ifndef MERLIN_INTERPOLANT_HPP_
#define MERLIN_INTERPOLANT_HPP_

#include "merlin/vector.hpp"  // merlin::floatvec
#include "merlin/nddata.hpp"  // merlin::NdData
#include "merlin/grid.hpp"  // merlin::CartesianGrid

namespace merlin {

class CartesianInterpolant {
  public:
    CartesianInterpolant(CartesianGrid & grid, NdData & value);
    ~CartesianInterpolant(void) = default;

    virtual float operator() (const floatvec & x);

  protected:
    CartesianGrid * grid_;
    NdData * value_;
};

class LagrangeInterpolant : public CartesianInterpolant {
  public:
    LagrangeInterpolant(CartesianGrid & grid, NdData & value);
    ~LagrangeInterpolant(void) = default;

  protected:
    /** @brief Interpolation coefficient vector.*/
    NdData coef_;
};

}  // namespace merlin

#endif  // MERLIN_INTERPOLANT_HPP_
