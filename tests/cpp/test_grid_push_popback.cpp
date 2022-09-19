// Copyright 2022 quocdang1998
#include <cstdio>

#include "merlin/grid.hpp"

int main (void) {
    merlin::RegularGrid gr(8,3);

    for (int i = 0; i < gr.size(); i++) {
        gr[i][{0}] = i*1.0;
        gr[i][{1}] = i*2.0;
        gr[i][{2}] = i*3.0;
    }

    MESSAGE("Pushback a point in the grid.\n");
    gr.push_back(merlin::Vector<float>({5.0, 10.0, 15.0}));
    MESSAGE("Popback a point in the grid.\n");
    gr.pop_back();

    MESSAGE("Expected 3 columns, col1 = range(0, 8), col2 = col1*2, col3 = col1*3.\n");
    for (merlin::RegularGrid::iterator it = gr.begin(); it != gr.end(); it++) {
        MESSAGE("Point number %I64u:", it.index()[0]);
        for (int j = 0; j < 3; j++) {
            std::printf("%4.1f", gr.grid_points()[it.index()]);
            if (j != 2) {
                it++;
                std::printf("  ");
            } else {
                std::printf("\n");
            }
        }
    }

}
