#include <cstdio>

#include "merlin/array.hpp"
#include "merlin/grid.hpp"

int main (void) {
    merlin::Grid gr(3,8);

    for (int i = 0; i < gr.npoint(); i++) {
        gr[i][{0}] = i*1.0;
        gr[i][{1}] = i*2.0;
        gr[i][{2}] = i*3.0;
    }

    gr.push_back(std::vector<float>({5.0, 10.0, 15.0}));

    for (merlin::Array::iterator it = gr.begin(); it != gr.end(); it++) {
        for (int j = 0; j < 3; j++) {
            std::printf("%.3f", gr.grid_points()[it.index()]);
            if (j != 2) {
                it++;
                std::printf("  ");
            } else {
                std::printf("\n");
            }
        }
    }

}