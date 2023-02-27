#include "merlin/array/array.hpp"
#include "merlin/array/slice.hpp"
#include "merlin/interpolant/cartesian_grid.hpp"
#include "merlin/interpolant/lagrange.hpp"
#include "merlin/interpolant/newton.hpp"
#include "merlin/logger.hpp"
#include "merlin/vector.hpp"

int main(void) {
    double data[9] = {1.0, 3.0, 9.0, 2.0, 4.0, 10.0, 2.5, 4.5, 10.5};
    std::uint64_t ndim = 2;
    std::uint64_t dims[2] = {3, 3};
    std::uint64_t strides[2] = {dims[1] * sizeof(double), sizeof(double)};
    merlin::array::Array value(data, ndim, dims, strides);
    std::printf("Initial array: %s\n", value.str().c_str());

    merlin::interpolant::CartesianGrid grid({{0.0, 1.0, 1.5}, {0.0, 1.0, 2.0}});
    merlin::Vector<merlin::array::Slice> slices(ndim);
    merlin::array::Array coeff(value.shape());
    merlin::interpolant::calc_lagrange_coeffs_cpu(grid, value, coeff);
    // merlin::interpolant::calc_newton_coeffs_cpu(grid, value, coeff);
    // merlin::interpolant::calc_lagrange_coeffs_cpu(grid, value, slices, coeff);

    for (merlin::array::Array::iterator it = coeff.begin(); it != coeff.end(); ++it) {
        MESSAGE("Coefficient of index (%d, %d) : %f.\n", int(it.index()[0]), int(it.index()[1]), coeff.get(it.index()));
    }
    MESSAGE("Theorical calculated values (C-contiguous order): -1/2 3 -5/2 1 -4 3.\n");

    merlin::Vector<merlin::array::Slice> vec_slice(2);
    merlin::Vector<double> p1({1.0, 2.0});  // on grid
    double p1_eval = merlin::interpolant::eval_lagrange_cpu(grid, coeff, p1);
    // double p1_eval = merlin::interpolant::eval_lagrange_cpu(grid, coeff, vec_slice, p1);
    MESSAGE("Evaluated value: %f.\n", p1_eval);
    MESSAGE("Expected value: 1.0 + 2*2.0 + 1 = 10.0.\n");
    merlin::Vector<double> p2({1.0, 0.5});  // 1st dim on grid
    double p2_eval = merlin::interpolant::eval_lagrange_cpu(grid, coeff, p2);
    // double p2_eval = merlin::interpolant::eval_lagrange_cpu(grid, coeff, vec_slice, p2);
    MESSAGE("Evaluated value: %f.\n", p2_eval);
    MESSAGE("Expected value: 1.0 + 2*0.5*0.5 + 1 = 2.5.0.\n");
    merlin::Vector<double> p3({0.5, 0.25});  // both dim off grid
    double p3_eval = merlin::interpolant::eval_lagrange_cpu(grid, coeff, p3);
    // double p3_eval = merlin::interpolant::eval_lagrange_cpu(grid, coeff, vec_slice, p3);
    MESSAGE("Evaluated value: %f.\n", p3_eval);
    MESSAGE("Expected value: 0.5 + 2*0.25*0.25 + 1 = 1.625.\n");
}
