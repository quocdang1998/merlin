#include <cinttypes>

#include "merlin/array/array.hpp"
#include "merlin/array/operation.hpp"
#include "merlin/array/slice.hpp"
#include "merlin/interpolant/cartesian_grid.hpp"
#include "merlin/interpolant/sparse_grid.hpp"
#include "merlin/interpolant/interpolant.hpp"
#include "merlin/logger.hpp"
#include "merlin/vector.hpp"

int main(void) {
    double data[9] = {1.0, 3.0, 5.0, 2.0, 4.0, 6.0, 3.0, 5.0, 7.0};
    merlin::intvec dims({3, 3});
    merlin::intvec strides = merlin::array::contiguous_strides(dims, sizeof(double));
    merlin::array::Array value(data, dims, strides);
    MESSAGE("Original value: %s\n", value.str().c_str());

    merlin::interpolant::SparseGrid grid({{0.0, 1.0, 2.0}, {0.0, 1.0, 2.0}}, 1, merlin::intvec({1, 1}));
    MESSAGE("Level index of sparse grid:\n");
    for (std::uint64_t level_index = 0; level_index < grid.num_level(); level_index++) {
        std::printf("    Level %" PRIu64 ": %s\n", level_index, grid.level_index(level_index).str().c_str());
    }
    MESSAGE("Subgrid start index: %s\n", grid.sub_grid_start_index().str().c_str());

    merlin::interpolant::PolynomialInterpolant plm_int(grid, value, merlin::interpolant::Method::Lagrange);
    MESSAGE("Coefficients are: %s\n", plm_int.get_coeff().str().c_str());

    merlin::interpolant::CartesianGrid c_grid({{0.0, 1.0, 2.0}, {0.0, 1.0, 2.0}});
    for (std::uint64_t i = 0; i < c_grid.size(); i ++) {
        merlin::Vector<double> point = c_grid[i];
        MESSAGE("Value calculated at %s is %.6f.\n", point.str().c_str(), plm_int(point));
    }
}
