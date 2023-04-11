// Copyright 2022 quocdang1998
#include "merlin/intpl/interpolant.hpp"

#include <bitset>     // std::bitset
#include <cinttypes>  // PRIu64

#include "merlin/array/array.hpp"           // merlin::array::Array
#include "merlin/array/nddata.hpp"          // merlin::array::NdData
#include "merlin/array/parcel.hpp"          // merlin::array::Parcel
#include "merlin/intpl/cartesian_grid.hpp"  // merlin::intpl::CartesianGrid
#include "merlin/intpl/lagrange.hpp"        // merlin::intpl::calc_lagrange_coeffs_cpu
#include "merlin/intpl/newton.hpp"          // merlin::intpl::calc_newton_coeffs_cpu
#include "merlin/intpl/sparse_grid.hpp"     // merlin::intpl::SparseGrid
#include "merlin/logger.hpp"                // FAILURE, merlin::not_implemented_error
#include "merlin/vector.hpp"                // merlin::intvec

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// PolynomialInterpolant
// ---------------------------------------------------------------------------------------------------------------------

// Constructor from a Cartesian grid and an array of values using CPU
intpl::PolynomialInterpolant::PolynomialInterpolant(const intpl::CartesianGrid & grid, const array::Array & values,
                                                    intpl::Method method) {
    // check for dimensionality of the grid and the values
    if (grid.ndim() != values.ndim()) {
        FAILURE(std::invalid_argument,
                "Expected grid and value array have the same dimension, got %" PRIu64 " (grid) and %" PRIu64
                " (data).\n",
                grid.ndim(), values.ndim());
    }
    std::uint64_t ndim = grid.ndim();
    intvec grid_shape = grid.get_grid_shape();
    for (std::uint64_t i_dim = 0; i_dim < ndim; i_dim++) {
        if (grid_shape[i_dim] != values.shape()[i_dim]) {
            FAILURE(std::invalid_argument,
                    "Cannot interpolate on grid and data with different dimensionality "
                    "(difference detected at dimension %" PRIu64 ").\n",
                    i_dim);
        }
    }
    this->grid_ = new intpl::CartesianGrid(grid);
    this->method_ = method;
    // calculate coefficients
    this->coeff_ = new array::Array(values.shape());
    array::Array * p_coeff_array = static_cast<array::Array *>(this->coeff_);
    if (method == intpl::Method::Lagrange) {
        intpl::calc_lagrange_coeffs_cpu(grid, values, *p_coeff_array);
    } else {
        intpl::calc_newton_coeffs_cpu(grid, values, *p_coeff_array);
    }
}

// Constructor from a Cartesian grid and an array of values using GPU
intpl::PolynomialInterpolant::PolynomialInterpolant(const intpl::CartesianGrid & grid, const array::Parcel & values,
                                                    intpl::Method method, const cuda::Stream & stream,
                                                    std::uint64_t n_threads) {
    // check for dimensionality of the grid and the values
    if (grid.ndim() != values.ndim()) {
        FAILURE(std::invalid_argument,
                "Expected grid and value array have the same dimension, got %" PRIu64 " (grid) and %" PRIu64
                " (data).\n",
                grid.ndim(), values.ndim());
    }
    std::uint64_t ndim = grid.ndim();
    intvec grid_shape = grid.get_grid_shape();
    for (std::uint64_t i_dim = 0; i_dim < ndim; i_dim++) {
        if (grid_shape[i_dim] != values.shape()[i_dim]) {
            FAILURE(std::invalid_argument,
                    "Cannot interpolate on grid and data with different dimensionality "
                    "(difference detected at dimension %" PRIu64 ").\n",
                    i_dim);
        }
    }
    this->grid_ = new intpl::CartesianGrid(grid);
    this->method_ = method;
    // calculate coefficients
    this->coeff_ = new array::Parcel(values.shape(), stream);
    array::Parcel * p_coeff_array = static_cast<array::Parcel *>(this->coeff_);
    if (method == intpl::Method::Lagrange) {
        intpl::calc_lagrange_coeffs_gpu(grid, values, *p_coeff_array, stream, n_threads);
    } else {
        intpl::calc_newton_coeffs_gpu(grid, values, *p_coeff_array, stream, n_threads);
    }
}

// Constructor from Sparse grid and a flattened array of values
intpl::PolynomialInterpolant::PolynomialInterpolant(const intpl::SparseGrid & grid, const array::Array & values,
                                                    intpl::Method method) {
    // check for dimensionality of the grid and the values
    if (grid.ndim() != values.ndim()) {
        FAILURE(std::invalid_argument,
                "Expected grid and value array have the same dimension, got %" PRIu64 " (grid) and %" PRIu64
                " (data).\n",
                grid.ndim(), values.ndim());
    }
    std::uint64_t ndim = grid.ndim();
    intvec grid_shape = grid.get_grid_shape();
    for (std::uint64_t i_dim = 0; i_dim < ndim; i_dim++) {
        if (grid_shape[i_dim] != values.shape()[i_dim]) {
            FAILURE(std::invalid_argument,
                    "Cannot interpolate on grid and data with different dimensionality "
                    "(difference detected at dimension %" PRIu64 ").\n",
                    i_dim);
        }
    }
    this->grid_ = new intpl::SparseGrid(grid);
    this->method_ = method;
    // calculate coefficients
    this->coeff_ = new array::Array(intvec({grid.size()}));
    array::Array * p_coeff_array = static_cast<array::Array *>(this->coeff_);
    if (method == intpl::Method::Lagrange) {
        intpl::calc_lagrange_coeffs_cpu(grid, values, *p_coeff_array);
    } else {
        intpl::calc_newton_coeffs_cpu(grid, values, *p_coeff_array);
    }
}

// Copy constructor
intpl::PolynomialInterpolant::PolynomialInterpolant(const intpl::PolynomialInterpolant & src) {
    // copy old grid to new grid
    if (src.is_grid_cartesian()) {
        this->grid_ = new intpl::CartesianGrid(*static_cast<const intpl::CartesianGrid *>(src.grid_));
    } else {
        this->grid_ = new intpl::SparseGrid(*static_cast<const intpl::SparseGrid *>(src.grid_));
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
intpl::PolynomialInterpolant & intpl::PolynomialInterpolant::operator=(const intpl::PolynomialInterpolant & src) {
    // delete old data
    if (this->grid_ != nullptr) {
        delete this->grid_;
    }
    if (this->coeff_ != nullptr) {
        delete this->coeff_;
    }
    // copy old grid to new grid
    if (src.is_grid_cartesian()) {
        this->grid_ = new intpl::CartesianGrid(*static_cast<const intpl::CartesianGrid *>(src.grid_));
    } else {
        this->grid_ = new intpl::SparseGrid(*static_cast<const intpl::SparseGrid *>(src.grid_));
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
bool intpl::PolynomialInterpolant::is_calc_on_cpu(void) const {
    if (const array::Array * p_coeff = dynamic_cast<const array::Array *>(this->coeff_); p_coeff != nullptr) {
        return true;
    } else if (const array::Parcel * p_coeff = dynamic_cast<const array::Parcel *>(this->coeff_); p_coeff == nullptr) {
        FAILURE(std::invalid_argument, "Cannot determine the processor of calculation.\n");
    }
    return false;
}

// Get grid type
bool intpl::PolynomialInterpolant::is_grid_cartesian(void) const {
    // shorten the name
    using CartGrid = intpl::CartesianGrid;
    using SparseGrid = intpl::SparseGrid;
    if (const CartGrid * p_grid = dynamic_cast<const CartGrid *>(this->grid_); p_grid != nullptr) {
        return true;
    } else if (const SparseGrid * p_grid = dynamic_cast<const SparseGrid *>(this->grid_); p_grid == nullptr) {
        FAILURE(std::invalid_argument, "Cannot determine the type of the grid.\n");
    }
    return false;
}

// Evaluate interpolation at a point
double intpl::PolynomialInterpolant::operator()(const Vector<double> & point) const {
    // get status (1st bit indicates CPU/GPU, 2nd bit indicates Cart/Sparse, 3rd bit indicates Lagrange/Newton)
    std::bitset<2> status = 0;
    if (!(this->is_calc_on_cpu())) {
        FAILURE(std::invalid_argument, "Single point evaluation not supported on GPU.\n");
    }
    if (this->is_grid_cartesian()) {
        status.reset(1);
    } else {
        status.set(1);
    }
    if (this->method_ == intpl::Method::Lagrange) {
        status.reset(0);
    } else {
        status.set(0);
    }
    // declare variables for switch statement
    const intpl::CartesianGrid * p_cart_grid;
    const intpl::SparseGrid * p_sparse_grid;
    const array::Array * p_array_coeff;
    const array::Parcel * p_parcel_coeff;
    // for each case
    switch (status.to_ulong()) {
    case 0 :  // CPU for Cart grid, Lagrange
        p_cart_grid = static_cast<const intpl::CartesianGrid *>(this->grid_);
        p_array_coeff = static_cast<const array::Array *>(this->coeff_);
        if (point.size() != p_cart_grid->ndim()) {
            FAILURE(std::invalid_argument, "Expected point with dimension %" PRIu64 ", got %" PRIu64 ".\n",
                    p_cart_grid->ndim(), point.size());
        }
        return intpl::eval_lagrange_cpu(*p_cart_grid, *p_array_coeff, point);
        break;
    case 1 :  // CPU for Cart grid, Newton
        p_cart_grid = static_cast<const intpl::CartesianGrid *>(this->grid_);
        p_array_coeff = static_cast<const array::Array *>(this->coeff_);
        if (point.size() != p_cart_grid->ndim()) {
            FAILURE(std::invalid_argument, "Expected point with dimension %" PRIu64 ", got %" PRIu64 ".\n",
                    p_cart_grid->ndim(), point.size());
        }
        return intpl::eval_newton_cpu(*p_cart_grid, *p_array_coeff, point);
        break;
    case 2 :  // CPU for Sparse grid, Lagrange
        p_sparse_grid = static_cast<const intpl::SparseGrid *>(this->grid_);
        p_array_coeff = static_cast<const array::Array *>(this->coeff_);
        if (point.size() != p_sparse_grid->ndim()) {
            FAILURE(std::invalid_argument, "Expected point with dimension %" PRIu64 ", got %" PRIu64 ".\n",
                    p_sparse_grid->ndim(), point.size());
        }
        return intpl::eval_lagrange_cpu(*p_sparse_grid, *p_array_coeff, point);
        break;
    case 3 :  // CPU for Sparse grid, Newton
        p_sparse_grid = static_cast<const intpl::SparseGrid *>(this->grid_);
        p_array_coeff = static_cast<const array::Array *>(this->coeff_);
        if (point.size() != p_sparse_grid->ndim()) {
            FAILURE(std::invalid_argument, "Expected point with dimension %" PRIu64 ", got %" PRIu64 ".\n",
                    p_sparse_grid->ndim(), point.size());
        }
        return intpl::eval_newton_cpu(*p_sparse_grid, *p_array_coeff, point);
        break;
    default :  // not implemented error
        FAILURE(not_implemented_error, "Case not yet implemented.\n");
    }
    return 0.0;
}

// Evaluate interpolation at multiple points on GPU
Vector<double> intpl::PolynomialInterpolant::operator()(const array::Parcel & points, const cuda::Stream & stream,
                                                        std::uint64_t n_thread) const {
    if (!(this->is_grid_cartesian()) || this->is_calc_on_cpu()) {
        FAILURE(std::invalid_argument, "Cannot use this function for sparse grid or CPU constructed coefficients.\n");
    }
    const intpl::CartesianGrid * p_grid = static_cast<const intpl::CartesianGrid *>(this->grid_);
    const array::Parcel * p_coeff = static_cast<const array::Parcel *>(this->coeff_);
    if (this->method_ == intpl::Method::Lagrange) {
        return intpl::eval_lagrange_gpu(*p_grid, *p_coeff, points, stream, n_thread);
    }
    return intpl::eval_newton_gpu(*p_grid, *p_coeff, points, stream, n_thread);
}

// Destructor
intpl::PolynomialInterpolant::~PolynomialInterpolant(void) {
    if (this->grid_ != nullptr) {
        delete this->grid_;
    }
    if (this->coeff_ != nullptr) {
        delete this->coeff_;
    }
}

}  // namespace merlin
