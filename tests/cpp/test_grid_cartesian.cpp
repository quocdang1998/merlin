// Copyright 2022 quocdang1998
#include <cinttypes>

#include "merlin/array/array.hpp"
#include "merlin/grid/cartesian_grid.hpp"
#include "merlin/logger.hpp"

int main(void) {
    using namespace merlin;

    grid::CartesianGrid cart_gr({{0.1, 0.2, 0.3}, {1.0, 3.0, 2.0, 4.0}, {0.0, 0.25}});
    MESSAGE("Cartesian grid: %s\n", cart_gr.str().c_str());
    MESSAGE("Number of nodes: %" PRIu64 ", number of points: %" PRIu64 "\n", cart_gr.num_nodes(), cart_gr.size());

    grid::CartesianGrid cart_gr2(cart_gr);
    MESSAGE("Copied Cartesian grid: %s\n", cart_gr.str().c_str());

    array::Array grid_points = cart_gr2.get_points();
    MESSAGE("Grid points: %s\n", grid_points.str().c_str());
}
