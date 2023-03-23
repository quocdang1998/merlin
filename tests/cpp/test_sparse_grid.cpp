#include <cinttypes>
#include <cstdint>

#include "merlin/interpolant/sparse_grid.hpp"
#include "merlin/logger.hpp"
#include "merlin/vector.hpp"

int main(void) {
    merlin::Vector<double> v1 = {0, 0.25, 0.5, 0.75, 1.0};
    merlin::Vector<double> v2 = {0, 0.5, 1.0};
    merlin::Vector<double> v3 = {0.5};
    merlin::interpolant::SparseGrid grid({v1, v2, v3});
    std::uint64_t num_level_vector = grid.num_level();
    for (int i = 0; i < num_level_vector; i++) {
        MESSAGE("Level vector %d: %s.\n", i, grid.level_index(i).str().c_str());
        MESSAGE("Index of first point in sub-grid relative to the grid: %" PRIu64 ".\n",
                grid.sub_grid_start_index()[i]);
    }
    MESSAGE("Number of points in grid: %" PRIu64 ".\n", grid.size());

    merlin::interpolant::SparseGrid ani_grid({v1, v2, v3}, 3, {2, 1, 1});
    num_level_vector = ani_grid.num_level();
    for (int i = 0; i < num_level_vector; i++) {
        MESSAGE("Anisotropic grid's level vector %d: %s.\n", i, ani_grid.level_index(i).str().c_str());
    }
}
