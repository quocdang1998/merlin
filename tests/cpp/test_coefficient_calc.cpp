#include "merlin/array/array.hpp"
#include "merlin/interpolant/cartesian_grid.hpp"
#include "merlin/interpolant/interpolant.hpp"
#include "merlin/logger.hpp"
#include "merlin/vector.hpp"

int main(void) {
    float data[6] = {1.0, 3.0, 5.0, 2.0, 4.0, 6.0};
    std::uint64_t ndim = 2;
    std::uint64_t dims[2] = {2, 3};
    std::uint64_t strides[2] = {dims[1] * sizeof(float), sizeof(float)};
    merlin::array::Array value(data, ndim, dims, strides);

    merlin::interpolant::CartesianGrid grid({{0.0, 1.0}, {0.0, 1.0, 2.0}});
    merlin::Vector<merlin::array::Slice> slices(ndim);
    merlin::array::Array coeff = merlin::calc_lagrange_coeffs_cpu(&grid, &value, slices);

    for (merlin::array::Array::iterator it = coeff.begin(); it != coeff.end(); ++it) {
        MESSAGE("Coefficient of index (%d, %d) : %f.\n", int(it.index()[0]), int(it.index()[1]), coeff.get(it.index()));
    }
    MESSAGE("Theorical calculated values (C-contiguous order): -1/2 3 -5/2 1 -4 3.\n");

    merlin::floatvec p1({1.0, 2.0});  // on grid
    float p1_eval = merlin::eval_lagrange_cpu(&grid, &coeff, p1);
    MESSAGE("Evaluated value: %f.\n", p1_eval);
    merlin::floatvec p2({1.0, 0.5});  // 1st dim on grid
    float p2_eval = merlin::eval_lagrange_cpu(&grid, &coeff, p2);
    MESSAGE("Evaluated value: %f.\n", p2_eval);
    merlin::floatvec p3({0.5, 0.25});  // both dim off grid
    float p3_eval = merlin::eval_lagrange_cpu(&grid, &coeff, p3);
    MESSAGE("Evaluated value: %f.\n", p3_eval);
}
