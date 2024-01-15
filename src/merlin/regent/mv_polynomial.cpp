// Copyright 2024 quocdang1998
#include "merlin/regent/mv_polynomial.hpp"

#include "merlin/utils.hpp"  // merlin::prod_elements

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// MvPolynomial
// ---------------------------------------------------------------------------------------------------------------------

// Constructor of an empty polynomial from order per dimension
regent::MvPolynomial::MvPolynomial(const intvec & order_per_dim) :
order_(order_per_dim), coeff_(prod_elements(order_per_dim)) { }

// Constructor of a pre-allocated array of coefficients and order per dimension
regent::MvPolynomial::MvPolynomial(double * coeff_data, const intvec & order_per_dim) : order_(order_per_dim) {
    this->coeff_.assign(coeff_data, prod_elements(order_per_dim));
}

}  // namespace merlin
