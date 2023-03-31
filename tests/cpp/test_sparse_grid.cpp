#include <cinttypes>
#include <cstdint>
#include <iostream>

#include "merlin/interpolant/sparse_grid.hpp"
#include "merlin/logger.hpp"
#include "merlin/vector.hpp"

int main(void) {
    merlin::Vector<double> br = {0, 300, 600, 900, 1200};  // l = 2
    merlin::Vector<double> pm = {157};  // l = 0
    merlin::Vector<double> tf = {500, 560, 600, 700, 800, 900, 1000, 1100, 1200};  // l = 3
    merlin::Vector<double> tm = {300, 400, 500, 560, 600};  // l = 2
    merlin::Vector<double> xe = {0.1, 0.4, 0.9};  // l = 1

    merlin::interpolant::SparseGrid grid({br, pm, tf, tm, xe}, 6, {1, 1, 1, 1, 1});
    std::cout << grid.str() << "\n";
}
