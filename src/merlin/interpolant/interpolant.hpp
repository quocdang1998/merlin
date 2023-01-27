// Copyright 2022 quocdang1998
#ifndef MERLIN_INTERPOLANT_INTERPOLANT_HPP_
#define MERLIN_INTERPOLANT_INTERPOLANT_HPP_

#include "merlin/array/nddata.hpp"  // merlin::array::NdData
#include "merlin/array/array.hpp"  // merlin::array::Array
#include "merlin/interpolant/grid.hpp"  // merlin::CartesianGrid
#include "merlin/vector.hpp"  // merlin::floatvec, merlin::Vector

namespace merlin::interpolant {
class CartesianInterpolant;
class LagrangianInterpolant;
}  // namespace merlin::interpolant

namespace merlin {

class interpolant::CartesianInterpolant {
  public:
    /** @brief Method to choose.*/
    enum class Method {
        Lagrange,
        Newton
    };

    CartesianInterpolant(const interpolant::CartesianGrid & grid, const array::NdData & value,
                         array::NdData & coeff, interpolant::CartesianInterpolant::Method method);
    ~CartesianInterpolant(void);

    float operator()(const floatvec & x) {return 0.0;}

  protected:
    const interpolant::CartesianGrid * pgrid_ = nullptr;
    const array::NdData * pvalue_ = nullptr;
    array::NdData * pcoeff_ = nullptr;
};

float eval_lagrange_cpu(const interpolant::CartesianGrid * pgrid, const array::NdData * pcoeff,
                        const merlin::floatvec & x);

void calc_lagrange_coeffs_cpu(const interpolant::CartesianGrid * pgrid, const array::Array * pvalue,
                                      const Vector<array::Slice> & slices, array::Array * presult);

array::Array calc_lagrange_coeffs_cpu(const interpolant::SparseGrid * pgrid, const array::Array * pvalue);

array::Array calc_lagrange_coeffs_gpu(const interpolant::CartesianGrid * pgrid, const array::Array * pvalue,
                                      const Vector<array::Slice> & slices,
                                      const cuda::Stream & stream = cuda::Stream());

}  // namespace merlin

#endif  // MERLIN_INTERPOLANT_INTERPOLANT_HPP_
