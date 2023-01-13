#include <cinttypes>
#include <cstdint>

#include "merlin/interpolant/sparse_grid.hpp"
#include "merlin/logger.hpp"
#include "merlin/vector.hpp"

int main(void) {
    merlin::floatvec v1 = {0, 0.25, 0.5, 0.75, 1.0};
    merlin::floatvec v2 = {0, 0.5, 1.0};
    merlin::floatvec v3 = {0.5};
    merlin::interpolant::SparseGrid grid({v1, v1, v1});
    std::uint64_t num_level_vector = grid.num_subgrid();
    for (int i = 0; i < num_level_vector; i++) {
        MESSAGE("Level vector %d: (%" PRIu64 ", %" PRIu64 ", %" PRIu64 ").\n", i,
                grid.level_vectors()[i*grid.ndim()+0],
                grid.level_vectors()[i*grid.ndim()+1],
                grid.level_vectors()[i*grid.ndim()+2]);
        MESSAGE("Index of first point in sub-grid relative to the grid: %" PRIu64 ".\n",
                grid.sub_grid_start_index()[i]);
    }
    MESSAGE("Number of points in grid: %" PRIu64 ".\n", grid.size());

    merlin::interpolant::SparseGrid ani_grid({v1, v2, v3}, 3, {2, 1, 1});
    num_level_vector = ani_grid.num_subgrid();
    for (int i = 0; i < num_level_vector; i++) {
        MESSAGE("Anisotropic grid's level vector %d: (%" PRIu64 ", %" PRIu64 ", %" PRIu64 ").\n", i,
                ani_grid.level_vectors()[i*ani_grid.ndim()+0],
                ani_grid.level_vectors()[i*ani_grid.ndim()+1],
                ani_grid.level_vectors()[i*ani_grid.ndim()+2]);
    }
}
