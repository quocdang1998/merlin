// Copyright 2022 quocdang1998
#ifndef MERLIN_INTERPOLANT_INTERPOLANT_HPP_
#define MERLIN_INTERPOLANT_INTERPOLANT_HPP_

#include "merlin/array/declaration.hpp"  // merlin::array::Array, merlin::array::Slice
#include "merlin/interpolant/grid.hpp"  // merlin::interpolant::CartesianGrid
#include "merlin/vector.hpp"  // merlin::Vector

namespace merlin::interpolant {
class PolynomialInterpolant;
class CartesianInterpolant;  // interpolant on a cartesian grid
}  // namespace merlin::interpolant

namespace merlin {

/** @brief Basic interpolant class.*/
class interpolant::PolynomialInterpolant {
  public:
    /** @brief Interpolant method.*/
    enum class Method : unsigned int {
        Lagrange,
        Newton
    };
    /** @brief Processor.*/
    enum class Processor : unsigned int {
        Cpu,
        Gpu
    };

    PolynomialInterpolant(void) {}
    virtual double operator()(const Vector<double> & point) {return 0.0;}
    virtual ~PolynomialInterpolant(void) {}
};

class interpolant::CartesianInterpolant : public interpolant::PolynomialInterpolant {
  public:
    /** @brief Default constructor.*/
    CartesianInterpolant(void) {}
    /** @brief Contructor from grid and value.
     *  @param grid Full Cartesian grid.
     *  @param value Values of function to interpolate at points on the grid. The array may contains all points in the
     *  grid, or just part of it. In the second case, the argument ``slices`` will indicate how that part was sliced.
     *  @param slices Slices indicating on which point of the grid the value array is on.
     *  @param method Method of interpoaltion to choose from.
     */
    CartesianInterpolant(const CartesianGrid & grid, const array::Array & value, const Vector<array::Slice> & slices,
                         interpolant::PolynomialInterpolant::Method method);

    /** @brief Default destructor.*/
    ~CartesianInterpolant(void) {}

  protected:
    /** @brief Grid.*/
    interpolant::CartesianGrid grid_;
    /** @brief Interpolation coefficient.*/
    array::Array coeff_;
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
