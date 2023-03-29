// Copyright 2022 quocdang1998
#include <cinttypes>

#include "merlin/array/array.hpp"
#include "merlin/interpolant/cartesian_grid.hpp"
#include "merlin/vector.hpp"
#include "merlin/logger.hpp"

int main(void) {
    merlin::Vector<double> v1 = {0.1, 0.2, 0.3};
    merlin::Vector<double> v2 = {1.0, 2.0, 3.0, 4.0};
    merlin::Vector<double> v3 = {0.0, 0.25};
    merlin::interpolant::CartesianGrid cart_gr({v1, v2, v3});

    MESSAGE("Loop with begin/end iterator.\n");
    for (merlin::interpolant::CartesianGrid::iterator it = cart_gr.begin(); it != cart_gr.end(); it++) {
        merlin::Vector<double> point = cart_gr[it.index()];
        MESSAGE("%f %f %f\n", point.data()[0], point.data()[1], point.data()[2]);
    }

    MESSAGE("Loop with C-contiguous index.\n");
    for (int i = 0; i < cart_gr.size(); i++) {
        merlin::Vector<double> point = cart_gr[i];
        MESSAGE("%f %f %f\n", point.data()[0], point.data()[1], point.data()[2]);
    }

    merlin::array::Array ar = cart_gr.grid_points();

    merlin::interpolant::CartesianGrid gr_1({{1.0, 2.0}, {0.3, 0.4, 0.5}});
    merlin::interpolant::CartesianGrid gr_2({{3.0, 4.0}, {0.1, 0.2, 0.4}});
    gr_1 += gr_2;
    MESSAGE("Vector of union:\n");
    for (int i = 0; i < gr_1.ndim(); i++) {
        const merlin::Vector<double> & v = gr_1.grid_vectors()[i];
        std::printf("%s", v.str().c_str());
        std:printf("\n");
    }
}
