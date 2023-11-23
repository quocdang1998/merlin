#include <numeric>
#include <iostream>
#include <vector>

#include "merlin/array/array.hpp"
#include "merlin/splint/cartesian_grid.hpp"

#include "spgrid/sparse_grid.hpp"
#include "spgrid/utils.hpp"

double foo(const merlin::floatvec & v) {
    return (v[0] + v[1] / v[2]) * v[3];
}

int main(void) {
    merlin::floatvec br = {0, 300, 600, 900, 1200};  // l = 2
    merlin::floatvec tf = {500, 560, 600, 700, 800, 900, 1000, 1100, 1200};  // l = 3
    merlin::floatvec tm = {300, 400, 500, 560, 600};  // l = 2
    merlin::floatvec xe = {0.1, 0.4, 0.9};  // l = 1

    // test constructor by predicate
    spgrid::SparseGrid grid(
        merlin::Vector<merlin::floatvec>({br, tf, tm, xe}),
        [] (const merlin::intvec & level) { return std::accumulate(level.begin(), level.end(), 0) < 6; }
    );
    std::cout << grid.str() << "\n";
/*
    // test constructor by level vectors (pass)
    spgrid::SparseGrid grid2(
        merlin::Vector<merlin::floatvec>({br, tf, tm, xe}),
        merlin::intvec({1, 1, 2, 1, 1, 2, 0, 0, 1, 2, 0, 1, 1, 2, 1, 0, 1, 2, 1, 1, 1, 3, 0, 1})
    );
    std::cout << grid2.str() << "\n";
*/
/*
    // test copy from full grid array (pass)
    merlin::array::Array full_data(grid.shape());
    for (std::uint64_t i = 0; i < grid.fullgrid().size(); i++) {
        full_data.set(i, foo(grid.fullgrid()[i]));
    }
    merlin::floatvec copied_data(spgrid::copy_sparsegrid_data_from_cartesian(full_data, grid1));
    std::cout << copied_data.str() << "\n";
*/
    // test get Cartesian grid for each array
    std::vector<char> buffer(grid.fullgrid().cumalloc_size());
    for (std::uint64_t i_level = 0; i_level < grid.nlevel(); i_level++) {
        std::cout << "Level " << i_level << " " << grid.get_ndlevel_at_index(i_level).str();
        std::cout << ": " << grid.get_grid_at_level(i_level, buffer.data()).str() << "\n";
    }
}
