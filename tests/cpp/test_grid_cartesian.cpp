// Copyright 2022 quocdang1998

#include "merlin/array/array.hpp"
#include "merlin/grid/cartesian_grid.hpp"
#include "merlin/logger.hpp"

using namespace merlin;

int main(void) {
    grid::CartesianGrid cart_gr({{0.1, 0.2, 0.3}, {1.0, 3.0, 2.0, 4.0}, {0.0, 0.25}});
    Message("Cartesian grid: {}\n", cart_gr.str());
    Message("Number of nodes: {}, number of points: {}\n", cart_gr.num_nodes(), cart_gr.size());

    grid::CartesianGrid cart_gr2(cart_gr);
    Message("Copied Cartesian grid: {}\n", cart_gr.str());

    array::Array grid_points = cart_gr2.get_points();
    Message("Grid points: ") << grid_points.str() << "\n";
}
