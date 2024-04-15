// Copyright 2024 quocdang1998
#include "merlin/regpl/vandermonde.hpp"

#include <numeric>  // std::iota

#include <omp.h>  // ::omp_get_thread_num

#include "merlin/array/array.hpp"          // merlin::array::Array
#include "merlin/grid/cartesian_grid.hpp"  // merlin::grid::CartesianGrid
#include "merlin/grid/regular_grid.hpp"    // merlin::grid::RegularGrid
#include "merlin/logger.hpp"               // merlin::Fatal
#include "merlin/regpl/polynomial.hpp"     // merlin::regpl::Polynomial
#include "merlin/utils.hpp"                // merlin::prod_elements, merlin::contiguous_to_ndim_idx

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------------------------------------------------

// Calculate a row of Vandermonde matrix
void vandermonde_entry(std::uint64_t * power, double * point, const std::uint64_t & ndim, double & dest) noexcept {
    dest = 1.0;
    for (std::uint64_t i_dim = 0; i_dim < ndim; i_dim++) {
        for (std::uint64_t j = 0; j < power[i_dim]; j++) {
            dest *= point[i_dim];
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Vandermonde
// ---------------------------------------------------------------------------------------------------------------------

// Constructor from a full polynomial and Cartesian grid
regpl::Vandermonde::Vandermonde(const Index & order, const grid::CartesianGrid & grid, std::uint64_t n_threads) :
order_(order), term_idx_(prod_elements(order.data(), grid.ndim())) {
    // create term index vector
    std::iota(this->term_idx_.begin(), this->term_idx_.end(), 0);
    // create Vandermonde matrix
    this->solver_ = linalg::QRPDecomp(grid.size(), this->term_idx_.size());
    _Pragma("omp parallel num_threads(n_threads)") {
        std::uint64_t thread_idx = ::omp_get_thread_num();
        Index power;
        power.fill(0);
        Point point;
        point.fill(0);
        for (std::uint64_t i_term = thread_idx; i_term < this->term_idx_.size(); i_term += n_threads) {
            // get power per dimension of the term
            contiguous_to_ndim_idx(i_term, order.data(), grid.ndim(), power.data());
            // calculate vandermonde matrix for each point in the grid
            for (std::uint64_t i_point = 0; i_point < grid.size(); i_point++) {
                grid.get(i_point, point.data());
                vandermonde_entry(power.data(), point.data(), grid.ndim(), this->solver_.core().get(i_point, i_term));
            }
        }
    }
    // decompose the matrix
    this->solver_.decompose(n_threads);
}

// Constructor from a full polynomial and grid points
regpl::Vandermonde::Vandermonde(const Index & order, const grid::RegularGrid & grid, std::uint64_t n_threads) :
order_(order), term_idx_(prod_elements(order.data(), grid.ndim())) {
    // create term index vector
    std::iota(this->term_idx_.begin(), this->term_idx_.end(), 0);
    // create Vandermonde matrix
    this->solver_ = linalg::QRPDecomp(grid.size(), this->term_idx_.size());
    _Pragma("omp parallel num_threads(n_threads)") {
        std::uint64_t thread_idx = ::omp_get_thread_num();
        Index power;
        power.fill(0);
        Point point;
        point.fill(0);
        for (std::uint64_t i_term = thread_idx; i_term < this->term_idx_.size(); i_term += n_threads) {
            // get power per dimension of the term
            contiguous_to_ndim_idx(i_term, order.data(), grid.ndim(), power.data());
            // calculate vandermonde matrix for each point in the grid
            for (std::uint64_t i_point = 0; i_point < grid.size(); i_point++) {
                grid.get(i_point, point.data());
                vandermonde_entry(power.data(), point.data(), grid.ndim(), this->solver_.core().get(i_point, i_term));
            }
        }
    }
    // decompose the matrix
    this->solver_.decompose(n_threads);
}

// Solve for the coefficients given the data
regpl::Polynomial regpl::Vandermonde::solve(DoubleVec & values_to_fit) const {
    // check argument
    if (values_to_fit.size() != this->solver_.nrow()) {
        Fatal<std::invalid_argument>(
            "Data must have the same number of points as the grid used to construct the Vandermonde matrix.\n");
    }
    // solve for the coefficients
    this->solver_.solve(values_to_fit.data());
    // construct the result polynomial
    DoubleVec coeff;
    coeff.assign(values_to_fit.data(), this->term_idx_.size());
    return regpl::Polynomial(coeff, this->order_, this->term_idx_);
}

}  // namespace merlin
