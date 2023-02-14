#include <cinttypes>

#include "merlin/array/array.hpp"
#include "merlin/array/slice.hpp"
#include "merlin/interpolant/sparse_grid.hpp"
#include "merlin/interpolant/lagrange.hpp"
#include "merlin/logger.hpp"
#include "merlin/vector.hpp"

int main(void) {
    double data[9] = {1.0, 3.0, 5.0, 2.0, 4.0, 6.0, 3.0, 5.0, 7.0};
    std::uint64_t ndim = 2;
    std::uint64_t dims[2] = {3, 3};
    std::uint64_t strides[2] = {dims[1] * sizeof(double), sizeof(double)};
    merlin::array::Array value(data, ndim, dims, strides);

    merlin::interpolant::SparseGrid grid({{0.0, 1.0, 2.0}, {0.0, 1.0, 2.0}}, 1, merlin::intvec({1, 1}));
    merlin::array::Array coeff(merlin::intvec({grid.size()}));
    merlin::interpolant::copy_value_from_cartesian_array(coeff, value, grid);
    merlin::interpolant::calc_lagrange_coeffs_cpu(grid, coeff);
    MESSAGE("After calculation\n");
    for (std::uint64_t i = 0; i < coeff.shape()[0]; i++) {
        MESSAGE("Coefficient %" PRIu64 " %.5f.\n", i, coeff.get(i));
    }

    double f_x = merlin::interpolant::eval_lagrange_cpu(grid, coeff, {1.0, 1.0});
    MESSAGE("Value calculated at (0.0, 2.0) is %.6f.\n", f_x);
}
