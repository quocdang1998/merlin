// Copyright 2022 quocdang1998
#include "merlin/interpolant/interpolant.hpp"

#include <cinttypes>  // PRIu64
#include <bitset>  // std::bitset

#include "merlin/array/array.hpp"  // merlin::array::Array
#include "merlin/array/nddata.hpp"  // merlin::array::NdData
#include "merlin/array/parcel.hpp"  // merlin::array::Parcel
#include "merlin/interpolant/cartesian_grid.hpp"  // merlin::interpolant::CartesianGrid
#include "merlin/interpolant/lagrange.hpp"  // merlin::interpolant::calc_lagrange_coeffs_cpu
#include "merlin/interpolant/newton.hpp"  // merlin::interpolant::calc_newton_coeffs_cpu
#include "merlin/interpolant/sparse_grid.hpp"  // merlin::interpolant::SparseGrid
#include "merlin/logger.hpp"  // FAILURE, merlin::not_implemented_error
#include "merlin/vector.hpp"  // merlin::intvec

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// PolynomialInterpolant
// --------------------------------------------------------------------------------------------------------------------

// Constructor from a Cartesian grid and an array of values using CPU
interpolant::PolynomialInterpolant::PolynomialInterpolant(const interpolant::CartesianGrid & grid,
                                                          const array::Array & values, interpolant::Method method) {
    // check for dimensionality of the grid and the values
    if (grid.ndim() != values.ndim()) {
        FAILURE(std::invalid_argument, "Expected grid and value array have the same dimension, got %" PRIu64
                " (grid) and %" PRIu64 " (data).\n", grid.ndim(), values.ndim());
    }
    std::uint64_t ndim = grid.ndim();
    intvec grid_shape = grid.get_grid_shape();
    for (std::uint64_t i_dim = 0; i_dim < ndim; i_dim++) {
        if (grid_shape[i_dim] != values.shape()[i_dim]) {
            FAILURE(std::invalid_argument, "Cannot interpolate on grid and data with different dimensionality "
                    "(difference detected at dimension %" PRIu64 ").\n", i_dim);
        }
    }
    this->grid_ = new interpolant::CartesianGrid(grid);
    this->method_ = method;
    // calculate coefficients
    this->coeff_ = new array::Array(values.shape());
    array::Array * p_coeff_array = static_cast<array::Array *>(this->coeff_);
    if (method == interpolant::Method::Lagrange) {
        interpolant::calc_lagrange_coeffs_cpu(grid, values, *p_coeff_array);
    } else {
        interpolant::calc_newton_coeffs_cpu(grid, values, *p_coeff_array);
    }
}

// Constructor from a Cartesian grid and an array of values using GPU
interpolant::PolynomialInterpolant::PolynomialInterpolant(const interpolant::CartesianGrid & grid,
                                                          const array::Parcel & values, interpolant::Method method,
                                                          const cuda::Stream & stream, std::uint64_t n_threads) {
    // check for dimensionality of the grid and the values
    if (grid.ndim() != values.ndim()) {
        FAILURE(std::invalid_argument, "Expected grid and value array have the same dimension, got %" PRIu64
                " (grid) and %" PRIu64 " (data).\n", grid.ndim(), values.ndim());
    }
    std::uint64_t ndim = grid.ndim();
    intvec grid_shape = grid.get_grid_shape();
    for (std::uint64_t i_dim = 0; i_dim < ndim; i_dim++) {
        if (grid_shape[i_dim] != values.shape()[i_dim]) {
            FAILURE(std::invalid_argument, "Cannot interpolate on grid and data with different dimensionality "
                    "(difference detected at dimension %" PRIu64 ").\n", i_dim);
        }
    }
    this->grid_ = new interpolant::CartesianGrid(grid);
    this->method_ = method;
    // calculate coefficients
    this->coeff_ = new array::Parcel(values.shape());
    array::Parcel * p_coeff_array = static_cast<array::Parcel *>(this->coeff_);
    if (method == interpolant::Method::Lagrange) {
        interpolant::calc_lagrange_coeffs_gpu(grid, values, *p_coeff_array, stream, n_threads);
    } else {
        interpolant::calc_newton_coeffs_gpu(grid, values, *p_coeff_array, stream, n_threads);
    }
}

// Constructor from Sparse grid and a flattened array of values
interpolant::PolynomialInterpolant::PolynomialInterpolant(const interpolant::SparseGrid & grid,
                                                          const array::Array & values, interpolant::Method method) {
    // check for dimensionality of the grid and the values
    if (grid.ndim() != values.ndim()) {
        FAILURE(std::invalid_argument, "Expected grid and value array have the same dimension, got %" PRIu64
                " (grid) and %" PRIu64 " (data).\n", grid.ndim(), values.ndim());
    }
    std::uint64_t ndim = grid.ndim();
    intvec grid_shape = grid.get_grid_shape();
    for (std::uint64_t i_dim = 0; i_dim < ndim; i_dim++) {
        if (grid_shape[i_dim] != values.shape()[i_dim]) {
            FAILURE(std::invalid_argument, "Cannot interpolate on grid and data with different dimensionality "
                    "(difference detected at dimension %" PRIu64 ").\n", i_dim);
        }
    }
    this->grid_ = new interpolant::SparseGrid(grid);
    this->method_ = method;
    // calculate coefficients
    this->coeff_ = new array::Array(intvec({grid.size()}));
    array::Array * p_coeff_array = static_cast<array::Array *>(this->coeff_);
    if (method == interpolant::Method::Lagrange) {
        interpolant::calc_lagrange_coeffs_cpu(grid, values, *p_coeff_array);
    } else {
        interpolant::calc_newton_coeffs_cpu(grid, values, *p_coeff_array);
    }
}

// Copy constructor
interpolant::PolynomialInterpolant::PolynomialInterpolant(const interpolant::PolynomialInterpolant & src) {
    // copy old grid to new grid
    if (src.is_grid_cartesian()) {
        this->grid_ = new interpolant::CartesianGrid(*static_cast<const interpolant::CartesianGrid *>(src.grid_));
    } else {
        this->grid_ = new interpolant::SparseGrid(*static_cast<const interpolant::SparseGrid *>(src.grid_));
    }
    // copy old array of coeff to new array of coeff
    if (src.is_calc_on_cpu()) {
        this->coeff_ = new array::Array(*static_cast<const array::Array *>(src.coeff_));
    } else {
        this->coeff_ = new array::Parcel(*static_cast<const array::Parcel *>(src.coeff_));
    }
    // copy method
    this->method_ = src.method_;
}

// Copy assignment
interpolant::PolynomialInterpolant & interpolant::PolynomialInterpolant::operator=(const interpolant::PolynomialInterpolant & src) {
    // delete old data
    if (this->grid_ != nullptr) {
        delete this->grid_;
    }
    if (this->coeff_ != nullptr) {
        delete this->coeff_;
    }
    // copy old grid to new grid
    if (src.is_grid_cartesian()) {
        this->grid_ = new interpolant::CartesianGrid(*static_cast<const interpolant::CartesianGrid *>(src.grid_));
    } else {
        this->grid_ = new interpolant::SparseGrid(*static_cast<const interpolant::SparseGrid *>(src.grid_));
    }
    // copy old array of coeff to new array of coeff
    if (src.is_calc_on_cpu()) {
        this->coeff_ = new array::Array(*static_cast<const array::Array *>(src.coeff_));
    } else {
        this->coeff_ = new array::Parcel(*static_cast<const array::Parcel *>(src.coeff_));
    }
    // copy method
    this->method_ = src.method_;
    return *this;
}

// Get processor
bool interpolant::PolynomialInterpolant::is_calc_on_cpu(void) const {
    if (const array::Array * p_coeff = dynamic_cast<const array::Array *>(this->coeff_); p_coeff != nullptr) {
        return true;
    } else if (const array::Parcel * p_coeff = dynamic_cast<const array::Parcel *>(this->coeff_); p_coeff == nullptr) {
        FAILURE(std::invalid_argument, "Cannot determine the processor of calculation.\n");
    }
    return false;
}

// Get grid type
bool interpolant::PolynomialInterpolant::is_grid_cartesian(void) const {
    // shorten the name
    using CartGrid = interpolant::CartesianGrid;
    using SparseGrid = interpolant::SparseGrid;
    if (const CartGrid * p_grid = dynamic_cast<const CartGrid *>(this->grid_); p_grid != nullptr) {
        return true;
    } else if (const SparseGrid * p_grid = dynamic_cast<const SparseGrid *>(this->grid_); p_grid == nullptr) {
        FAILURE(std::invalid_argument, "Cannot determine the type of the grid.\n");
    }
    return false;
}

// Evaluate interpolation at a point
double interpolant::PolynomialInterpolant::operator()(const Vector<double> & point) const {
    // get status (1st bit indicates CPU/GPU, 2nd bit indicates Cart/Sparse, 3rd bit indicates Lagrange/Newton)
    std::bitset<2> status = 0;
    if (!(this->is_calc_on_cpu())) {
        FAILURE(std::invalid_argument, "Single point evaluation not supported on GPU.\n");
    }
    if (this->is_grid_cartesian()) {status.reset(1);} else {status.set(1);}
    if (this->method_ == interpolant::Method::Lagrange) {status.reset(0);} else {status.set(0);}
    // declare variables for switch statement
    const interpolant::CartesianGrid * p_cart_grid;
    const interpolant::SparseGrid * p_sparse_grid;
    const array::Array * p_array_coeff;
    const array::Parcel * p_parcel_coeff;
    // for each case
    switch (status.to_ulong()) {
    case 0:  // CPU for Cart grid, Lagrange
        p_cart_grid = static_cast<const interpolant::CartesianGrid *>(this->grid_);
        p_array_coeff = static_cast<const array::Array *>(this->coeff_);
        if (point.size() != p_cart_grid->ndim()) {
            FAILURE(std::invalid_argument, "Expected point with dimension %" PRIu64 ", got %" PRIu64 ".\n",
                    p_cart_grid->ndim(), point.size());
        }
        return interpolant::eval_lagrange_cpu(*p_cart_grid, *p_array_coeff, point);
        break;
    case 1:  // CPU for Cart grid, Newton
        p_cart_grid = static_cast<const interpolant::CartesianGrid *>(this->grid_);
        p_array_coeff = static_cast<const array::Array *>(this->coeff_);
        if (point.size() != p_cart_grid->ndim()) {
            FAILURE(std::invalid_argument, "Expected point with dimension %" PRIu64 ", got %" PRIu64 ".\n",
                    p_cart_grid->ndim(), point.size());
        }
        return interpolant::eval_newton_cpu(*p_cart_grid, *p_array_coeff, point);
        break;
    case 2:  // CPU for Sparse grid, Lagrange
        p_sparse_grid = static_cast<const interpolant::SparseGrid *>(this->grid_);
        p_array_coeff = static_cast<const array::Array *>(this->coeff_);
        if (point.size() != p_sparse_grid->ndim()) {
            FAILURE(std::invalid_argument, "Expected point with dimension %" PRIu64 ", got %" PRIu64 ".\n",
                    p_sparse_grid->ndim(), point.size());
        }
        return interpolant::eval_lagrange_cpu(*p_sparse_grid, *p_array_coeff, point);
        break;
    case 3:  // CPU for Sparse grid, Newton
        p_sparse_grid = static_cast<const interpolant::SparseGrid *>(this->grid_);
        p_array_coeff = static_cast<const array::Array *>(this->coeff_);
        if (point.size() != p_sparse_grid->ndim()) {
            FAILURE(std::invalid_argument, "Expected point with dimension %" PRIu64 ", got %" PRIu64 ".\n",
                    p_sparse_grid->ndim(), point.size());
        }
        return interpolant::eval_newton_cpu(*p_sparse_grid, *p_array_coeff, point);
        break;
    default:  // not implemented error
        FAILURE(not_implemented_error, "Case not yet implemented.\n");
    }
    return 0.0;
}

// Evaluate interpolation at multiple points on GPU
Vector<double> interpolant::PolynomialInterpolant::operator()(const array::Parcel & points,
                                                              const cuda::Stream & stream,
                                                              std::uint64_t n_thread) const {
    if (!(this->is_grid_cartesian()) || this->is_calc_on_cpu()) {
        FAILURE(std::invalid_argument, "Cannot use this function for sparse grid or CPU constructed coefficients.\n");
    }
    const interpolant::CartesianGrid * p_grid = static_cast<const interpolant::CartesianGrid *>(this->grid_);
    const array::Parcel * p_coeff = static_cast<const array::Parcel *>(this->coeff_);
    if (this->method_ == interpolant::Method::Lagrange) {
        return interpolant::eval_lagrange_gpu(*p_grid, *p_coeff, points, stream, n_thread);
    }
    return interpolant::eval_newton_gpu(*p_grid, *p_coeff, points, stream, n_thread);
}

// Destructor
interpolant::PolynomialInterpolant::~PolynomialInterpolant(void) {
    if (this->grid_ != nullptr) {
        delete this->grid_;
    }
    if (this->coeff_ != nullptr) {
        delete this->coeff_;
    }
}

}  // namespace merlin
