#include <cstdio>

#include "merlin/array.hpp"
#include "merlin/grid.hpp"

int main (void) {
    merlin::Grid gr(3,5);

    for (int i = 0; i < gr.npoint(); i++) {
        gr[i][0] = i*1.0;
        gr[i][1] = i*2.0;
        gr[i][2] = i*3.0;
    }

    for (merlin::Array::iterator it = gr.begin(); it != gr.end(); it++) {
        std::printf("%.3f\n", gr.grid_points()[it.index()]);
    }

}