#include <cinttypes>

#include "merlin/interpolant/sparse_grid.hpp"
#include "merlin/logger.hpp"
#include "merlin/vector.hpp"

int main(void) {
    merlin::floatvec v1 = {0, 0.25, 0.5, 0.75, 1.0};
    merlin::floatvec v2 = {0, 0.5, 1.0};
    merlin::floatvec v3 = {0.5};
    merlin::interpolant::SparseGrid grid({v1, v1, v1});
    grid.calc_level_vectors();
    for (int i = 0; i < grid.level_vectors().size(); i++) {
        const merlin::intvec & level_vector = grid.level_vectors()[i];
        MESSAGE("Level vector %d: (%" PRIu64 ", %" PRIu64 ", %" PRIu64 ").\n", i, level_vector[0], level_vector[1], level_vector[2]);
    }

    merlin::interpolant::SparseGrid ani_grid({v1, v2, v3}, 2, {2, 1, 1});
    ani_grid.calc_level_vectors();
    for (int i = 0; i < ani_grid.level_vectors().size(); i++) {
        const merlin::intvec & level_vector = ani_grid.level_vectors()[i];
        MESSAGE("Anisotropic grid's level vector %d: (%" PRIu64 ", %" PRIu64 ", %" PRIu64 ").\n", i, level_vector[0], level_vector[1], level_vector[2]);
    }
}
