#include <algorithm>
#include <numeric>
#include <iostream>
#include <vector>

#include "merlin/array/array.hpp"
#include "merlin/grid/cartesian_grid.hpp"
#include "merlin/utils.hpp"

#include "spgrid/sparse_grid.hpp"
#include "spgrid/interpolator.hpp"
#include "spgrid/utils.hpp"

double foo(const merlin::floatvec & v) {
    return (v[0] + v[1] / v[2]) * v[3];
}

merlin::array::Array point_generator(std::uint64_t num_point, const merlin::grid::CartesianGrid & grid) {
    std::mt19937 gen;
    std::vector<std::uniform_real_distribution<double>> dists;
    dists.reserve(grid.ndim());
    for (std::uint64_t i_dim = 0; i_dim < grid.ndim(); i_dim++) {
        const merlin::floatvec grid_vector = grid.grid_vector(i_dim);
        const auto [it_min, it_max] = std::minmax_element(grid_vector.cbegin(), grid_vector.cend());
        dists.push_back(std::uniform_real_distribution<double>(*it_min, *it_max));
    }
    merlin::array::Array points({num_point, grid.ndim()});
    for (std::uint64_t i_point = 0; i_point < num_point; i_point++) {
        for (std::uint64_t i_dim = 0; i_dim < grid.ndim(); i_dim++) {
            points[{i_point, i_dim}] = dists[i_dim](gen);
        }
    }
    return points;
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
/*
    // test get Cartesian grid for each array
    std::vector<char> buffer(grid.fullgrid().cumalloc_size());
    for (std::uint64_t i_level = 0; i_level < grid.nlevel(); i_level++) {
        std::cout << "Level " << i_level << " " << grid.get_ndlevel_at_index(i_level).str();
        std::cout << ": " << grid.get_grid_at_level(i_level, buffer.data()).str() << "\n";
    }
*/
    // test Sparse grid interpolation
    merlin::array::Array value(grid.fullgrid().shape());
    for (std::uint64_t i = 0; i < grid.fullgrid().size(); i++) {
        merlin::intvec index(merlin::contiguous_to_ndim_idx(i, grid.fullgrid().shape()));
        value[index] = foo(grid.fullgrid()[index]);
    }
    merlin::Vector<merlin::splint::Method> methods = {
        merlin::splint::Method::Newton,
        merlin::splint::Method::Newton,
        merlin::splint::Method::Newton,
        merlin::splint::Method::Newton
    };
    spgrid::Interpolator interp(grid, value, methods, 10);
    // std::cout << "Coefficients: " << interp.get_coeff().str() << "\n";
    // double points_data[] = {150, 510, 350, 0.5, 1000, 1000, 550, 0.8};
    // merlin::array::Array points(points_data, {2, 4}, {4*sizeof(double), sizeof(double)}, false);
    merlin::array::Array points(point_generator(100, grid.fullgrid()));
    std::cout << "Points: " << points.str() << "\n";
    std::cout << "Evaluated: " << interp.evaluate(points, 5).str() << "\n";
    // std::cout << "Reference: " << foo({150, 510, 350, 0.5}) << " " << foo({1000, 1000, 550, 0.8}) << "\n";
}
