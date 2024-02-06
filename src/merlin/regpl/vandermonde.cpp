// Copyright 2024 quocdang1998
#include "merlin/regpl/vandermonde.hpp"

#include <numeric>  // std::iota

#include <omp.h>  // #pragma omp

#include "merlin/grid/cartesian_grid.hpp"  // merlin::grid::CartesianGrid
#include "merlin/logger.hpp"               // FAILURE
#include "merlin/regpl/polynomial.hpp"     // merlin::regpl::Polynomial
#include "merlin/utils.hpp"                // merlin::prod_elements, merlin::contiguous_to_ndim_idx

#include "Eigen/Core"   // Eigen::setNbThreads
#include "Eigen/Dense"  // Eigen::MatrixXd, Eigen::Map, Eigen::VectorXd
#include "Eigen/SVD"    // Eigen::BDCSVD

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------------------------------------------------

// Alias
using EigenSvd = Eigen::BDCSVD<Eigen::MatrixXd, Eigen::ComputeThinU | Eigen::ComputeThinV>;

// Calculate an entry of Vandermonde matrix
void vandermonde_entry(std::uint64_t * term, double * point, const std::uint64_t & ndim, double & dest) noexcept {
    dest = 1.0;
    for (std::uint64_t i_dim = 0; i_dim < ndim; i_dim++) {
        for (std::uint64_t j = 0; j < term[i_dim]; j++) {
            dest *= point[i_dim];
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Vandermonde
// ---------------------------------------------------------------------------------------------------------------------

// Constructor from a full polynomial and Cartesian grid
regpl::Vandermonde::Vandermonde(const intvec & order, const grid::CartesianGrid & grid, std::uint64_t n_threads) :
order_(order), term_idx_(prod_elements(order)) {
    // check argument
    if (order.size() != grid.ndim()) {
        FAILURE(std::invalid_argument, "Ndim of order vector and grid must be the same.\n");
    }
    // create term index vector
    std::iota(this->term_idx_.begin(), this->term_idx_.end(), 0);
    // create Vandermonde matrix
    Eigen::MatrixXd vandermonde_matrix(grid.size(), prod_elements(order));
    intvec buffer(n_threads * order.size());
    floatvec point_data(n_threads * grid.ndim());
    #pragma omp parallel for num_threads(n_threads)
    for (std::int64_t i_order = 0; i_order < vandermonde_matrix.cols(); i_order++) {
        // get order per dimension of the term
        std::uint64_t * thread_buffer = buffer.data() + ::omp_get_thread_num() * order.size();
        contiguous_to_ndim_idx(i_order, order, thread_buffer);
        // loop for each point in the grid
        double * thread_point = point_data.data() + ::omp_get_thread_num() * order.size();
        for (std::uint64_t i_point = 0; i_point < grid.size(); i_point++) {
            grid.get(i_point, thread_point);
            vandermonde_entry(thread_buffer, thread_point, grid.ndim(), vandermonde_matrix(i_point, i_order));
        }
    }
    // solve matrix
    Eigen::setNbThreads(n_threads);
    this->svd_decomp_ = new EigenSvd(vandermonde_matrix);
    Eigen::setNbThreads(0);
}

// Constructor from a full polynomial and grid points
regpl::Vandermonde::Vandermonde(const intvec & order, const array::Array & grid_points, std::uint64_t n_threads) :
order_(order), term_idx_(prod_elements(order)) {
    // check argument
    if (grid_points.ndim() != 2) {
        FAILURE(std::invalid_argument, "Invalid grid_points argument.\n");
    }
    if (!grid_points.is_c_contiguous()) {
        FAILURE(std::invalid_argument, "Grid_points must be C-contiguous.\n");
    }
    if (order.size() != grid_points.shape()[1]) {
        FAILURE(std::invalid_argument, "Ndim of order vector and points must be the same.\n");
    }
    // create term index vector
    std::iota(this->term_idx_.begin(), this->term_idx_.end(), 0);
    // create Vandermonde matrix
    Eigen::MatrixXd vandermonde_matrix(grid_points.shape()[0], prod_elements(order));
    intvec buffer(n_threads * order.size());
    #pragma omp parallel for num_threads(n_threads)
    for (std::int64_t i_order = 0; i_order < vandermonde_matrix.cols(); i_order++) {
        // get order per dimension of the term
        std::uint64_t * thread_buffer = buffer.data() + ::omp_get_thread_num() * order.size();
        contiguous_to_ndim_idx(i_order, order, thread_buffer);
        // loop for each point in the grid
        for (std::uint64_t i_point = 0; i_point < grid_points.shape()[0]; i_point++) {
            double * point_data = grid_points.data() + i_point * grid_points.shape()[1];
            vandermonde_entry(thread_buffer, point_data, grid_points.shape()[1], vandermonde_matrix(i_point, i_order));
        }
    }
    // solve matrix
    Eigen::setNbThreads(n_threads);
    this->svd_decomp_ = new EigenSvd(vandermonde_matrix);
    Eigen::setNbThreads(0);
}

// Solve for the coefficients given the data
regpl::Polynomial regpl::Vandermonde::solve(const floatvec & data, std::uint64_t n_threads) const {
    // cast pointer back and check argument
    if (this->svd_decomp_ == nullptr) {
        FAILURE(std::runtime_error, "Object not initialized.\n");
    }
    const EigenSvd * svd_solver = reinterpret_cast<EigenSvd *>(this->svd_decomp_);
    if (data.size() != svd_solver->rows()) {
        FAILURE(std::invalid_argument, "Data not having the correct number of points.\n");
    }
    // solve for the coefficients
    Eigen::Map<Eigen::VectorXd> data_eigen(const_cast<double *>(data.data()), data.size());
    Eigen::setNbThreads(n_threads);
    Eigen::VectorXd coeff_eigen = svd_solver->solve(data_eigen);
    Eigen::setNbThreads(0);
    floatvec coeff;
    coeff.assign(coeff_eigen.data(), coeff_eigen.size());
    return regpl::Polynomial(coeff, this->order_, this->term_idx_);
}

// Default destructor
regpl::Vandermonde::~Vandermonde(void) {
    if (this->svd_decomp_ != nullptr) {
        delete reinterpret_cast<EigenSvd *>(this->svd_decomp_);
    }
}

}  // namespace merlin
