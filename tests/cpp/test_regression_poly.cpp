#include <cmath>

#include <omp.h>

#include "merlin/array/array.hpp"
#include "merlin/array/operation.hpp"
#include "merlin/grid/cartesian_grid.hpp"
#include "merlin/grid/regular_grid.hpp"
#include "merlin/linalg/matrix.hpp"
#include "merlin/logger.hpp"
#include "merlin/regpl/polynomial.hpp"
#include "merlin/regpl/vandermonde.hpp"
// #include "merlin/regpl/core.hpp"
// #include "merlin/regpl/regressor.hpp"
#include "merlin/utils.hpp"

using namespace merlin;

void write_matrix_row(const floatvec & point, const intvec & order, double * row, std::uint64_t n_terms) {
    for (std::uint64_t i_term = 0; i_term < n_terms; i_term++) {
        row[i_term] = 1.0;
        intvec term_order = contiguous_to_ndim_idx(i_term, order);
        for (std::uint64_t i_dim = 0; i_dim < order.size(); i_dim++) {
            row[i_term] *= std::pow(point[i_dim], term_order[i_dim]);
        }
    }
}

floatvec reference(const grid::CartesianGrid & grid, const array::Array & data, const regpl::Polynomial & polynom) {
    // create matrix
    floatvec matrix_data(grid.size() * polynom.size());
    for (std::uint64_t i_point = 0; i_point < grid.size(); i_point++) {
        floatvec point = grid[i_point];
        write_matrix_row(point, polynom.order(), matrix_data.data() + i_point * polynom.size(), polynom.size());
    }
    linalg::Matrix matrix(matrix_data.data(), {grid.size(), polynom.size()});
    MESSAGE("Reference Vandermonde: %s\n", matrix.str().c_str());
    // multiply matrix by data
    floatvec result(polynom.size());
    return result;
}

int main(void) {
    // initialize data
    grid::CartesianGrid grid({{0.1, 0.2, 0.3}, {1.0, 2.0, 3.0, 4.0}, {0.0, 0.25}});
    double array_data[24] = {
        1.0, 2.4, 2.3, 5.1, 4.3, 2.7, 8.1, 9.1, 1.4, 2.7,
        1.1, 2.6, 2.1, 5.4, 4.5, 2.1, 8.4, 9.6, 1.3, 2.4
    };
    grid::RegularGrid rgrid(grid.get_points());
    array::Array data(array_data, grid.shape(), array::contiguous_strides(grid.shape(), sizeof(double)), false);
    floatvec data_vec;
    data_vec.assign(array_data, 24);
    regpl::Polynomial polynom({2, 3, 2});

    // reference
    floatvec ref_result = reference(grid, data, polynom);
    regpl::Vandermonde val(polynom.order(), grid, 2);
    regpl::Polynomial regressed_poly = val.solve(data_vec, 5);
    MESSAGE("Regressed polynomial: %s\n", regressed_poly.str().c_str());
}
