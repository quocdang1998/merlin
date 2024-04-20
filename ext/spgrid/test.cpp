#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "merlin/array/array.hpp"
#include "merlin/array/parcel.hpp"
#include "merlin/grid/cartesian_grid.hpp"
#include "merlin/utils.hpp"

#include "spgrid/hier_interpolator.hpp"
#include "spgrid/sparse_grid.hpp"
#include "spgrid/utils.hpp"

using namespace merlin;
using namespace spgrid;

double foo(const DoubleVec & v) { return (v[0] + v[1] / v[2]) * v[3]; }

array::Array point_generator(std::uint64_t num_point, const grid::CartesianGrid & grid) {
    std::mt19937 gen;
    std::vector<std::uniform_real_distribution<double>> dists;
    dists.reserve(grid.ndim());
    for (std::uint64_t i_dim = 0; i_dim < grid.ndim(); i_dim++) {
        const DoubleVec grid_vector = grid.grid_vector(i_dim);
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
    DoubleVec br = {0, 300, 600, 900, 1200};                          // l = 2
    DoubleVec tf = {500, 560, 600, 700, 800, 900, 1000, 1100, 1200};  // l = 3
    DoubleVec tm = {300, 400, 500, 560, 600};                         // l = 2
    DoubleVec xe = {0.1, 0.4, 0.9};                                   // l = 1

    // test constructor by predicate
    SparseGrid grid(Vector<DoubleVec>({br, tf, tm, xe}), [](const UIntVec & level) {
        bool first = std::accumulate(level.begin(), level.end(), std::uint64_t(0)) < 6;
        bool second = std::all_of(level.begin(), level.end(), [](const std::uint64_t & x) { return x < 2; });
        bool third = (level[1] == 0);
        bool fourth = (level[0] == 0);
        return first && second && third && fourth;
    });
    std::cout << grid.str() << "\n";

    // test constructor by level vectors (pass)
    SparseGrid grid2(Vector<DoubleVec>({br, tf, tm, xe}),
                     UIntVec({1, 1, 2, 1, 1, 2, 0, 0, 1, 2, 0, 1, 1, 2, 1, 0, 1, 2, 1, 1, 1, 3, 0, 1}));
    std::cout << grid2.str() << "\n";

    // test level iterator
    for (auto it = grid.begin(); it != grid.end(); ++it) {
        std::cout << it.str() << "\n";
        std::cout << get_grid(grid.fullgrid(), it.cum_idx).str() << "\n";
    }
    std::cout << "Finish\n";

    // test copy from full grid array (pass)
    array::Array full_data(grid.shape());
    for (std::uint64_t i = 0; i < grid.fullgrid().size(); i++) {
        full_data.set(i, foo(grid.fullgrid()[i]));
    }

    // test Sparse grid interpolation
    Vector<splint::Method> methods = {splint::Method::Newton, splint::Method::Newton, splint::Method::Newton,
                                      splint::Method::Newton};
    HierInterpolator hintpl(grid, full_data, methods);
    array::Array points = point_generator(1000, grid.fullgrid());
    DoubleVec result(1000);
    hintpl.evaluate(points, result);
    std::cout << result.str() << "\n";
}
