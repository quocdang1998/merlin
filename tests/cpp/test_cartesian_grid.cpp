// Copyright 2022 quocdang1998

#include "merlin/grid.hpp"
#include "merlin/vector.hpp"
#include "merlin/logger.hpp"

int main(void) {
    merlin::floatvec v1 = {0.1, 0.2, 0.3};
    merlin::floatvec v2 = {1.0, 2.0, 3.0, 4.0};
    merlin::floatvec v3 = {0.0, 0.25};
    merlin::CartesianGrid cart_gr = {v1, v2, v3};

/*
    MESSAGE("Loop with begin/end iterator.");
    for (merlin::Grid::iterator it = cart_gr.begin(); it != cart_gr.end(); it++) {
        merlin::Tensor ar = cart_gr[it.index()];
        MESSAGE("%f %f %f", ar.data()[0], ar.data()[1], ar.data()[2]);
    }

    MESSAGE("Loop with C-contiguous index.");
    for (unsigned int i = 0; i < cart_gr.npoint(); i++) {
        merlin::Tensor ar = cart_gr[i];
        MESSAGE("%f %f %f", ar.data()[0], ar.data()[1], ar.data()[2]);
    }

    merlin::Tensor ar = cart_gr.grid_points();
*/
}